from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


text_fields = {('glue', 'sst2'): ('sentence', None),
               ('glue', 'mrpc'): ('sentence1', 'sentence2'),
               ('glue', 'rte'):  ('sentence1', 'sentence2'),
               ('glue', 'qnli'): ('question', 'sentence'),
               ('glue', 'mnli'): ('premise', 'hypothesis'),
               ('glue', 'qqp'):  ('sentence1', 'sentence2')
}


def collate_fn_(batch, tokenizer, dataset_name, dataset_subname):
    is_float = isinstance(batch[0]['label'], list)
    batch = {key: [i[key] for i in batch] for key in batch[0]}
    sentence1_key, sentence2_key = text_fields[(dataset_name, dataset_subname)]   
    
    tokenized = tokenizer(batch.pop(sentence1_key), 
                          batch.pop(sentence2_key) if sentence2_key is not None else sentence2_key, 
                          padding=True, 
                          return_tensors='pt', 
                          return_token_type_ids=True)
    if is_float:
        tokenized.data['labels'] = torch.tensor(batch['label']).float()
    else:
        tokenized.data['labels'] = torch.tensor(batch['label']).long()
    return tokenized


def preprocess_data_for_asr(dataset, 
                            dataset_name, 
                            dataset_subname, 
                            tokenizer, 
                            target_model, 
                            batch_size=64, 
                            device='cpu'):    
    target_model.to(device)
    target_model.eval()
    
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=partial(collate_fn_, 
                                           tokenizer=tokenizer,
                                           dataset_name=dataset_name,
                                           dataset_subname=dataset_subname)
                        )
    lables = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_lables = target_model(**batch, return_dict=True).logits
            lables += batch_lables.softmax(dim=-1).cpu().tolist()
    mapped_dataset = dataset.map(lambda x, idx: {'label': lables[idx]}, with_indices=True)
    return mapped_dataset