from collections import defaultdict
from functools import partial
from heapq import nsmallest

import torch

from torch import nn
from torch.autograd.functional import jvp
from torch.autograd.functional import vjp
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from .utils import EarlyStop


class SimplexAttacker:
    def __init__(
        self,
        victim_model,
        target_model,
        q=2,
        attack_length=5,
        device=None,
        init_token_id=1996,
        initialization_mode="random",
        filtered_tokens_ids=[],
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = victim_model.to(self.device)
        self.target_model = target_model.to(self.device)

        self.q = q
        self.attack_length = attack_length

        self.vocab = self.model.vocab

        self.initialization_mode = initialization_mode
        self.filtered_tokens_ids = filtered_tokens_ids
        self.init_W()

        self.trigger_ids = torch.tensor([init_token_id] * self.attack_length).to(self.device)

        self.objective = torch.Tensor([float("inf")])
        self.results = []

    def init_W(self):
        if not hasattr(self, "W"):
            self.W = torch.ones(self.attack_length, 
                                self.vocab.shape[0], 
                                requires_grad=True).to(self.device)
        if self.initialization_mode == "random":
            self.W.exponential_()

        self.W[:, self.filtered_tokens_ids] = 0.0
        self.W /= self.W.sum(dim=-1, keepdim=True)

    @staticmethod
    def phi(x, p):
        return torch.sign(x) * torch.abs(x).pow(p - 1)

    def step(self, batch):
        batch = batch.to(self.device)

        inputs_embeds = self.model.get_inputs_embeds(batch["input_ids"])
        inputs_embeds[:, 1:self.attack_length + 1] = self.model.get_inputs_embeds(self.trigger_ids)

        f = partial(self.model.forward, **batch)

        x = torch.zeros_like(inputs_embeds)
        x[:, 1:self.attack_length + 1] = self.W @ self.vocab - self.model.get_inputs_embeds(self.trigger_ids)
        x = jvp(f, inputs_embeds, x)[1]
        x = self.phi(x, self.q)

        x = vjp(f, inputs_embeds, x)[1]
        x = x[:, 1:self.attack_length + 1].sum(dim=0) @ self.vocab.T
        return x

    def get_trigger(self):
        return self.trigger_ids

    def compute_logits(self, trigger_ids, batch):
        input_ids = batch["input_ids"]
        input_ids[:, 1 : self.attack_length + 1] = trigger_ids

        return self.target_model(
             attention_mask=batch["attention_mask"],
             token_type_ids=batch["token_type_ids"],
             input_ids=input_ids,
             return_dict=True,
         ).logits

    def compute_metric(self, trigger_ids, batches, metric_function):
        metric = 0.0

        with torch.no_grad():
            for batch, labels in batches:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                output = self.compute_logits(trigger_ids, batch)
                metric += metric_function(output, labels).cpu().numpy()
        return metric

    def compute_accuracy(self, trigger_ids):
        metric_function = lambda output, labels: (output.argmax(dim=-1) == labels).float().sum()
        return self.compute_metric(trigger_ids,
                                   self.eval_loader,
                                   metric_function) / len(self.eval_loader.dataset)

    def compute_batch_loss(self, trigger_ids, accum_batches):
        metric_function = lambda output, labels: cross_entropy(output, labels, reduction="sum")
        n_samples = sum(len(x) for x, _ in accum_batches)
        return -self.compute_metric(trigger_ids, 
                                    accum_batches, 
                                    metric_function) / n_samples

    def compute_batch_asr(self, trigger_ids, accum_batches):
        metric_function = lambda output, labels: (output.argmax(dim=-1) != labels).float().sum()
        n_samples = sum(len(x) for x, _ in accum_batches)
        return -self.compute_metric(trigger_ids, 
                                    accum_batches, 
                                    metric_function) / n_samples

    def beam_search(self, accum_batches, candidates, beam_size=1):
        current_triggers = [(self.objective, self.trigger_ids.cpu().clone())]
        trigger_ids = self.trigger_ids.clone()
        for token in candidates[0]:
            trigger_ids[0] = token
            criterion = self.compute_batch_asr(trigger_ids, accum_batches)
            current_triggers.append((criterion, trigger_ids.cpu().clone()))

        beam = nsmallest(beam_size, current_triggers, key=lambda x: x[0])
        for i in range(1, candidates.shape[0]):
            current_triggers = beam.copy()
            for _, trigger_ids in beam:
                for token in candidates[i]:
                    current_ids = trigger_ids.clone().to(self.device)
                    current_ids[i] = token
                    criterion = self.compute_batch_asr(current_ids, accum_batches)
                    current_triggers.append((criterion, current_ids.cpu().clone()))
            beam = nsmallest(beam_size, current_triggers, key=lambda x: x[0])

        return beam

    def train(
        self,
        train_loader,
        eval_loader=None,
        epochs=1,
        accumulation_steps=1,
        early_stop_patience=10,
        tokenizer=None,
        beam_size=1,
        topk=10,
    ):
        self.model.eval()
        self.accumulation_steps = accumulation_steps

        self.early_stop = EarlyStop(patience=early_stop_patience)

        self.train_loader = train_loader

        if eval_loader is None:
            self.eval_loader = train_loader
        else:
            self.eval_loader = eval_loader

        if self.accumulation_steps == -1:
            self.accumulation_steps = len(self.train_loader)

        for i in range(epochs):
            x, accum_batches = 0.0, []
            for j, (batch, labels) in enumerate(self.train_loader):
                accum_batches.append((batch, labels))
                batch_x = self.step(batch)
                x += batch_x

                if (j + 1) % self.accumulation_steps == 0:
                    x /= self.train_loader.batch_size * self.accumulation_steps
                    x[:, self.filtered_tokens_ids] = float("-inf")
                    x = torch.topk(x, k=topk, dim=-1, largest=True, sorted=True).indices

                    self.triggers = self.beam_search(accum_batches, x, beam_size)
                    self.trigger_ids = self.triggers[0][-1].to(self.device)

                    self.objective = self.compute_accuracy(self.trigger_ids)

                    current_step = j + i * len(self.train_loader)

                    print(
                        f"Iteration {current_step} / {len(self.train_loader) * epochs}, objective: {self.objective:.4f}",
                        end=" ",
                    )
                    if tokenizer is not None:
                        self.results.append(
                            {
                                "triggers": ", ".join(
                                    tokenizer.convert_ids_to_tokens(self.trigger_ids)
                                ),
                                "acc": self.objective,
                            }
                        )

                        print(tokenizer.convert_ids_to_tokens(self.trigger_ids))
                    else:
                        print()

                    self.init_W()
                    x, accum_batches = 0.0, []

                    if self.early_stop(-self.objective, self.trigger_ids):
                        self.trigger_ids = self.early_stop.best_triggers
                        return