import torch
from torch.nn.functional import cross_entropy
import numpy as np
import heapq
from operator import itemgetter
from copy import deepcopy

import sys
sys.path.append('../')
from src.utils import EarlyStop


def hotflip_attack(averaged_grad, embedding_matrix,
                   increase_loss=False, num_candidates=1, filtered_tokens_ids=[]):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.
    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()    
    
    averaged_grad = averaged_grad.unsqueeze(0)

    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                (averaged_grad, embedding_matrix))

    if len(filtered_tokens_ids) != 0:
        gradient_dot_embedding_matrix[:, :, filtered_tokens_ids] = -np.inf    
    
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    
    cand_trigger_token_ids = best_at_each_step[0].detach().cpu().numpy()
    if num_candidates == 1:
        cand_trigger_token_ids = cand_trigger_token_ids.reshape(-1, 1)
    
    return cand_trigger_token_ids


def move_to_device(batch, cuda_device):
    return {x:batch[x].to(cuda_device) for x in batch}


class WallaceAttack:
    """
    This is the Eric Wallace code adoptation (https://github.com/Eric-Wallace/universal-triggers).
    """
    def __init__(self, model, tokenizer, filtered_tokens_ids=[]):
        self.model = model
        self.tokenizer = tokenizer
        self.filtered_tokens_ids = filtered_tokens_ids
        
        self.orig_acc = 0.0
        self.triggers_set = set()
        self.embedding_weight = model.vocab
        self.embedding_weight.requires_grad=True
    
    
    def _accuracy(self, dev_dataset, trigger_token_ids):
        acc = 0.
        for batch in dev_dataset:
            with torch.no_grad():
                logits = self.evaluate_batch(batch, trigger_token_ids).logits
            acc += (logits.argmax(dim=-1) == batch['labels'].argmax(dim=-1).to(self.device)).float().sum().cpu() 
        if isinstance(dev_dataset, list):
            return acc / len(batch['labels'])
        else:
            return acc / len(dev_dataset.dataset)
    
    
    def get_accuracy(self, dev_dataset, trigger_token_ids=None):
        """
        When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
        triggers prepended for the whole dev_dataset.
        """

        self.model.eval() # model should be in eval() already, but just in case
        accuracy = self._accuracy(dev_dataset, trigger_token_ids).item()

        if trigger_token_ids is None:
            print("With 'the, the ...' Triggers: " + str(accuracy))
            self.orig_acc = accuracy
            self.triggers_set = set()
        else:
            print_string = ', '.join(self.tokenizer.convert_ids_to_tokens(trigger_token_ids))
            print("Current Triggers: " + print_string + " : " + str(accuracy))
            return print_string, accuracy
        
        
    def evaluate_batch(self, batch, trigger_token_ids=None, inputs_embeds=None):

        """
        Takes a batch of classification examples (SNLI or SST), and runs them through the model.
        If trigger_token_ids is not None, then it will append the tokens to the input.
        This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
        """
        batch = move_to_device(batch, cuda_device=self.device)
        if trigger_token_ids is None:
            output_dict = self.model(input_ids=batch['input_ids'], 
                                     token_type_ids=batch['token_type_ids'],
                                     attention_mask=batch['attention_mask'],
                                     labels=None,)
        else:
            attack_length = len(trigger_token_ids)
            trigger_sequence_tensor = torch.LongTensor(trigger_token_ids)
            trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['input_ids']), 1).to(self.device)

            input_ids = deepcopy(batch['input_ids'])
            input_ids[:, 1: attack_length + 1] = trigger_sequence_tensor

            output_dict = self.model(attention_mask=batch['attention_mask'],
                                     token_type_ids=batch['token_type_ids'],
                                     input_ids=input_ids,
                                     labels=None,
                                     inputs_embeds=inputs_embeds)
        output_dict.loss = cross_entropy(output_dict.logits, batch['labels'])
        return output_dict

        
    def get_average_grad(self, batch, trigger_token_ids):
        """
        Computes the average gradient w.r.t. the trigger tokens when prepended to every example
        in the batch. If target_label is set, that is used as the ground-truth label.
        """ 
        attack_length = len(trigger_token_ids)
        trigger_sequence_tensor = torch.LongTensor(trigger_token_ids)
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['input_ids']), 1).to(self.device)

        input_ids = deepcopy(batch['input_ids']).to(self.device)
        input_ids[:, 1: attack_length + 1] = trigger_sequence_tensor        

        embds = self.model.get_inputs_embeds(input_ids)
        
        loss = self.evaluate_batch(batch, trigger_token_ids, embds).loss
        grads = torch.autograd.grad(loss, embds)[0].cpu()

        # average grad across batch size, result only makes sense for trigger tokens at the front
        averaged_grad = torch.sum(grads, dim=0) 
        averaged_grad = averaged_grad[1:len(trigger_token_ids) + 1]# return just trigger grads
        return averaged_grad        
      
        
    def get_best_candidates(self, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
        """"
        Given the list of candidate trigger token ids (of number of trigger words by number of candidates
        per word), it finds the best new candidate trigger.
        This performs beam search in a left to right fashion.
        """
        # first round, no beams, just get the loss for each of the candidates in index 0.
        # (indices 1-end are just the old trigger)
        loss_per_candidate = self.get_loss_per_candidate(0, batch, trigger_token_ids,
                                                    cand_trigger_token_ids)
        # maximize the loss
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

        # top_candidates now contains beam_size trigger sequences, each with a different 0th token
        for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
            loss_per_candidate = []
            for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
                loss_per_candidate.extend(self.get_loss_per_candidate(idx, batch, cand, cand_trigger_token_ids))
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
        return max(top_candidates, key=itemgetter(1))[0]

    
    
    
    
    def get_loss_per_candidate(self, index, batch, trigger_token_ids, cand_trigger_token_ids):
        """
        For a particular index, the function tries all of the candidate tokens for that index.
        The function returns a list containing the candidate triggers it tried, along with their loss.
        """
        if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
            print("Only 1 candidate for index detected, not searching")
            return trigger_token_ids
        loss_per_candidate = []
        # loss for the trigger without trying the candidates
        curr_loss = -self._accuracy([batch], trigger_token_ids).item()
        
        loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
        for cand_id in range(len(cand_trigger_token_ids[0])):
            trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
            trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
            loss = -self._accuracy([batch], trigger_token_ids_one_replaced).item()
            loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
        return loss_per_candidate
        
        
    def train(self, val_loader, num_trigger_tokens=3, num_epochs=5, num_candidates=40, beam_size=1, device=None, patience=None):
        if patience is not None:
            early_stop = EarlyStop(patience)        
        if device is not None: self.device = device
        else: self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_accuracy(val_loader, trigger_token_ids=None)        
        self.model.eval()
        
        # initialize triggers which are concatenated to the input
        trigger_token_ids = [self.tokenizer.convert_tokens_to_ids('the')] * num_trigger_tokens

        # sample batches, update the triggers, and repeat
        results = []
        for epoch in range(num_epochs):
            for batch in val_loader:
                # get accuracy with current triggers
                triggers, objective = self.get_accuracy(val_loader, trigger_token_ids)
                result = {}
                result['triggers'], result['objective'] = triggers, objective
                results.append(result)
                
                if patience is not None:
                    if early_stop(-objective, triggers):
                        return results                

                # get gradient w.r.t. trigger embeddings for current batch}
                averaged_grad = self.get_average_grad(batch, trigger_token_ids)

                # pass the gradients to a particular attack to generate token candidates for each token.
                cand_trigger_token_ids = hotflip_attack(averaged_grad,
                                                        self.embedding_weight,
                                                        num_candidates=num_candidates,
                                                        increase_loss=True, filtered_tokens_ids=self.filtered_tokens_ids)
                
                # Tries all of the candidates and returns the trigger sequence with highest loss.
                trigger_token_ids = self.get_best_candidates(batch,
                                                             trigger_token_ids,
                                                             cand_trigger_token_ids,
                                                             beam_size=beam_size)

        # print accuracy after adding triggers
        triggers, objective = self.get_accuracy(val_loader, trigger_token_ids)
        result = {}
        result['triggers'], result['objective'] = triggers, objective
        results.append(result)
        return results 