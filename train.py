import argparse

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

from src.attacker import SimplexAttacker
from src.models import BertVictim, AlbertVictim

from torch.utils.data import DataLoader
from src.utils import collate_fn, insert_initial_trigger
from src.utils import preprocess_data_for_asr, set_seed

from src.utils import TokenFilter
from functools import partial

import os
import json
from transformers import AutoTokenizer
import numpy as np

import warnings   
warnings.simplefilter("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', help="q parameter", type=int, default=2)
    parser.add_argument('--layer', help="attacked layer", type=int, default=0)
    parser.add_argument('--beam_size', help="beam_size", type=int, default='1')
    parser.add_argument('--attack_length', help="attack_length", type=int, default=3)
    parser.add_argument('--topk', help="topk", type=int, default=10)
    parser.add_argument('--mode', help="how to init W", type=str, default='const')
    parser.add_argument('--early_stop_patience', help="early_stop_patience", type=int, default=10)
    parser.add_argument('--epochs', help="n epochs", type=int, default=50)
    parser.add_argument('--batch_size', help="batch_size", type=int, default=32)
    parser.add_argument('--accumulation_steps', help="accumulation_steps", type=int, default=4)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
    parser.add_argument('--checkpoint', help="dir wit models checkpoints", type=str, default='textattack/bert-base-uncased-MRPC')
    parser.add_argument('--dataset_name', help="dataset name", type=str, default='glue')
    parser.add_argument('--dataset_subname', help="dataset subname", type=str, default='mrpc')
    parser.add_argument('--dataset_split', help="dataset subname", type=str, default='validation')
    parser.add_argument('--results_dir', help="dir for results", type=str, default='./results')
    args = parser.parse_args()
    
    set_seed(args.seed)
    with open('task_to_keys.json', 'r') as f:
        task_to_keys = json.load(f)
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    dataset = load_dataset(args.dataset_name, args.dataset_subname)
    sentence1_key, sentence2_key = task_to_keys[args.dataset_subname]
            
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    victim_model = AutoModel.from_pretrained(args.checkpoint)
    target_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)            
    
    #albert's Victim model is different from bert's and roberta's one
    if 'albert' in args.checkpoint:
        victim_model = AlbertVictim(victim_model, layer=args.layer)
    else:
        victim_model = BertVictim(victim_model, layer=args.layer)
                  
    #make dataset with pseudolabels for fooling rate calculation
    preprocessed_dataset = preprocess_data_for_asr(dataset[args.dataset_split], 
                                                   sentence1_key, 
                                                   sentence2_key, 
                                                   tokenizer, 
                                                   target_model, 
                                                   batch_size=args.batch_size, 
                                                   device=args.device)
                
    #find id of init tocken 'the'
    init_token_id = tokenizer('the')['input_ids'][1]
                
    #tokens filter without fasttext's usage
    token_filter = TokenFilter(tokenizer=tokenizer)            

    #add three 'the' for each data sample in order to change them with triggers during attack training                
    trigger = ' '.join(['the'] * args.attack_length)
    train_dataset = preprocessed_dataset.map(partial(insert_initial_trigger, 
                                                     sapmle_part=sentence1_key, 
                                                     mode='front', 
                                                     trigger=trigger))
    #loader for training
    loader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        worker_init_fn=lambda x: np.random.seed(args.seed),
                        collate_fn=partial(collate_fn, 
                                           tokenizer=tokenizer, 
                                           sentence1_key=sentence1_key, 
                                           sentence2_key=sentence2_key, 
                                           train=False))            
    #loader for evaluation
    eval_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             worker_init_fn=lambda x: np.random.seed(args.seed),
                             collate_fn=partial(collate_fn, 
                                                tokenizer=tokenizer, 
                                                sentence1_key=sentence1_key, 
                                                sentence2_key=sentence2_key, 
                                                train=False))

    file_name = f'attack_l={args.layer}_q={args.q}_t={args.attack_length}_bs={args.beam_size}_topk={args.topk}_mode={args.mode}'
                            
    attacker = SimplexAttacker(q=args.q, 
                               victim_model=victim_model, 
                               target_model=target_model, 
                               attack_length=args.attack_length,
                               init_token_id=init_token_id,
                               filtered_tokens_ids=token_filter.get_filtered_tokens_ids(),
                               initialization_mode=args.mode,
                               device=args.device)

    attacker.train(epochs=args.epochs,
                   accumulation_steps=args.accumulation_steps,
                   early_stop_patience=args.early_stop_patience,
                   tokenizer=tokenizer,
                   train_loader=loader,
                   eval_loader=eval_loader,
                   beam_size=args.beam_size,
                   topk=args.topk)
    
    
    with open(f'{args.results_dir}/{file_name}.txt', 'w') as f:
        f.write(min(attacker.results, key=lambda x: x['acc'])['triggers'])