import re
import random
import torch

import numpy as np

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStop:
    def __init__(self, patience=0):
        self.patience = patience
        self.best_objective = None
        self.best_triggers = None
        self.counter = 0

    def __call__(self, objective, triggers):
        if self.best_objective is None:
            self.best_objective = objective
            self.best_triggers = triggers
            return False
        if self.best_objective < objective:
            self.best_objective = objective
            self.best_triggers = triggers
            self.counter = 0
        else:
            if self.patience - self.counter == 0:
                return True
            self.counter += 1
        return False


class TokenFilter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        #add all unused tokens
        self.filtered_tokens_ids = set(
            id for token, id in tokenizer.get_vocab().items() if "[unused" in token
        )
        #add all special tokens
        self.filtered_tokens_ids |= set(tokenizer.all_special_ids)
        #add all tokens from vocab which make is_included return False 
        self.filtered_tokens_ids |= set(
            id
            for token, id in self.tokenizer.get_vocab().items()
            if not self.is_included(token)
        )

    def get_filtered_tokens_ids(self):
        return list(self.filtered_tokens_ids)

    def is_included(self, token):
        if token in self.filtered_tokens_ids:
            return False
        
        #convert token to string, in bert case for word pieces '##' will be at the start of the string
        token = self.tokenizer.convert_tokens_to_string([token]).strip()
        
        # length of string before processing
        len_before = len(token)
        
        #shows difference between token's length before and after processing with regexp
        delta_len = 0
        
        # if we have bert tokenizer
        if 'roberta' not in self.tokenizer.name_or_path and 'albert' not in self.tokenizer.name_or_path:
            #and if we have word piece
            if token[:2] == '##' and len(token) != 2:
                # difference between token's length will be 2(valid '##' at start of word piece will be erased by regexp)
                delta_len = 2
        #substitute all not a letter and not a digit symbols with ''
        token = re.sub(r"\W+", "", token)
        #substitute all not an english letter and not a digit symbols with ''
        token = re.sub(r"[^A-z0-9]", "", token)
        #token became shorten if we erase invalid symbols on preprocessing 
        if len(token) + delta_len != len_before:
            return False
        return True


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def insert_initial_trigger(sample, sapmle_part, mode, trigger):
    if mode == "front":
        sample[sapmle_part] = f"{trigger} {sample[sapmle_part]}"
    else:
        sample[sapmle_part] = f"{sample[sapmle_part][:-1]} {trigger} {sample[sapmle_part][-1]}"
    return sample


def collate_fn(batch, tokenizer, sentence1_key, sentence2_key=None, train=True):
    batch = {key: [i[key] for i in batch] for key in batch[0]}

    tokenized = tokenizer(
        batch.pop(sentence1_key),
        batch.pop(sentence2_key) if sentence2_key is not None else sentence2_key,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )

    if train:
        return tokenized
    else:
        label = torch.tensor(batch["label"]).long()
        return tokenized, label


def preprocess_data_for_asr(dataset, 
                            sentence1_key, 
                            sentence2_key, 
                            tokenizer, 
                            target_model, 
                            batch_size=64, 
                            device='cpu'):

    target_model.to(device)
    target_model.eval()
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=partial(collate_fn, 
                                           tokenizer=tokenizer,
                                           sentence1_key=sentence1_key,
                                           sentence2_key=sentence2_key)
                        )
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_labels = target_model(**batch, return_dict=True).logits
            labels += batch_labels.argmax(dim=-1).cpu().tolist()
    mapped_dataset = dataset.map(lambda x, idx: {'label': labels[idx]}, with_indices=True)
    return mapped_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True