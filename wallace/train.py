import argparse
import json

import os
import pandas as pd
import numpy as np
from functools import partial
from transformers import AutoTokenizer

from utls.data import collate_fn_, preprocess_data_for_asr
from utls.models import WallaceVictimModel
from utls.attacker import WallaceAttack

from datasets import load_dataset
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from src.utils import TokenFilter, insert_initial_trigger, set_seed

import warnings   
warnings.simplefilter("ignore")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam_size', help="beam_size", type=int, default=1)
    parser.add_argument('--attack_length', help="attack_length", type=int, default=3)
    parser.add_argument('--topk', help="topk", type=int, default=10)
    parser.add_argument('--early_stop_patience', help="early_stop_patience", type=int, default=10)
    parser.add_argument('--epochs', help="n epochs", type=int, default=5)
    parser.add_argument('--batch_size', help="batch_size", type=int, default=128)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
    parser.add_argument('--checkpoint', help="dir wit models checkpoints", type=str, default='textattack/bert-base-uncased-MRPC')
    parser.add_argument('--dataset_name', help="dataset name", type=str, default='glue')
    parser.add_argument('--dataset_subname', help="dataset subname", type=str, default='mrpc')
    parser.add_argument('--dataset_split', help="dataset subname", type=str, default='validation')
    parser.add_argument('--results_dir', help="dir for results", type=str, default='./results')
    args = parser.parse_args()
    

    set_seed(args.seed)

    with open('../task_to_keys.json', 'r') as f:
        task_to_keys = json.load(f)
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)   

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = WallaceVictimModel(args.checkpoint)
    model.to(args.device)
    token_filter = TokenFilter(tokenizer=tokenizer)  
    attacker = WallaceAttack(model, 
                             tokenizer, 
                             filtered_tokens_ids=token_filter.get_filtered_tokens_ids())            

    collate_fn = lambda batch: collate_fn_(batch, tokenizer, args.dataset_name, args.dataset_subname)
    valid_dataset = load_dataset(args.dataset_name, args.dataset_subname, split=args.dataset_split)
    sentence1_key, sentence2_key = task_to_keys[args.dataset_subname]
    trigger = ' '.join(['the'] * args.attack_length)
    valid_dataset_mapped = preprocess_data_for_asr(valid_dataset, 
                                                   args.dataset_name, 
                                                   args.dataset_subname, 
                                                   tokenizer, 
                                                   model.model,
                                                   batch_size=args.batch_size, 
                                                   device=args.device)
    preprocessed_dataset = valid_dataset_mapped.map(partial(insert_initial_trigger, 
                                                            sapmle_part=sentence1_key, 
                                                            mode='front', 
                                                            trigger=trigger))    
    val_loader = DataLoader(preprocessed_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            drop_last=False, 
                            collate_fn=collate_fn, 
                            worker_init_fn=lambda x: np.random.seed(args.seed))

    file_name = f'attack_ntt={args.attack_length}_topk={args.topk}_bs={args.beam_size}'    
    results = attacker.train(val_loader, 
                             num_trigger_tokens=args.attack_length,
                             num_epochs=args.epochs, 
                             beam_size=args.beam_size,
                             num_candidates=args.topk, 
                             device=args.device, 
                             patience=args.early_stop_patience)
    
    with open(f'{args.results_dir}/{file_name}.txt', 'w') as f:
        f.write(min(results, key=lambda x: x['objective'])['triggers'])
