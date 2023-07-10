import torch
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score


def move_to_device(batch, cuda_device=None):
    res = {}
    for x in batch:
        res[x] = batch[x].to(cuda_device)
    return res


def compute_accuracy(model, tokenizer, test_loader, trigger, device, verbose=False):
    model.eval()
    attack_length = len(trigger)
    trigger_token_ids = [tokenizer.get_vocab()[x] for x in trigger]
    trigger_seq_tensor = torch.LongTensor(trigger_token_ids)

    preds = []
    lbls = []
    
    is_first_batch = True
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = move_to_device(batch, cuda_device=device)
            trigger_sequence_tensor =\
            trigger_seq_tensor.repeat(len(batch['input_ids']), 1).to(device)

            input_ids = deepcopy(batch['input_ids'])
            input_ids[:, 1: attack_length + 1] = trigger_seq_tensor

            output_dict = model(attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                input_ids=input_ids,
                                return_dict=True)
            
            preds.append(output_dict['logits'].argmax(dim=1).cpu().numpy().reshape(-1, 1))
            lbls.append(labels.numpy().reshape(-1, 1))            

            if verbose and is_first_batch:
                is_first_batch = False
                print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
                print(tokenizer.convert_ids_to_tokens(input_ids[0]))

    preds = np.vstack(preds)
    lbls = np.vstack(lbls)
    return accuracy_score(lbls, preds)
