import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from torch.utils.data import DataLoader
from src.utils import collate_fn, insert_initial_trigger
from src.utils import preprocess_data_for_asr, set_seed
from src.accuracy import compute_accuracy

from functools import partial

import json
import numpy as np
import warnings
warnings.simplefilter("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger', help="txt file with trigger", type=str, default='./trigger.txt')
    parser.add_argument('--batch_size', help="batch_size", type=int, default=32)
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
    
    dataset = load_dataset(args.dataset_name, args.dataset_subname)
    sentence1_key, sentence2_key = task_to_keys[args.dataset_subname]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    
    with open(f'{args.trigger}', 'r') as f:
        trigger = f.readline()
    trigger = trigger.split(', ')
    attack_length = len(trigger)
    
    dataset = load_dataset(args.dataset_name, args.dataset_subname)
    sentence1_key, sentence2_key = task_to_keys[args.dataset_subname]
    preprocessed_dataset = preprocess_data_for_asr(dataset[args.dataset_split], 
                                               sentence1_key, 
                                               sentence2_key, 
                                               tokenizer, 
                                               model, 
                                               batch_size=args.batch_size, device=args.device)
    
    #add three 'the' for each data sample in order to change them with triggers during attack training                
    the_trigger = ' '.join(['the'] * attack_length)
    train_dataset = preprocessed_dataset.map(partial(insert_initial_trigger, 
                                                     sapmle_part=sentence1_key, 
                                                     mode='front', 
                                                     trigger=the_trigger))
          
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
    
    accuracy = compute_accuracy(model, 
                                tokenizer, 
                                eval_loader, 
                                trigger, 
                                args.device, 
                                verbose=False)
    print('asr: ', 1 - accuracy)
