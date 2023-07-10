from transformers import AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn


class WallaceVictimModel(nn.Module):
    def __init__(self, model_chckpt):
        super().__init__()
        if isinstance(model_chckpt, list):
            config = AutoConfig.from_pretrained(model_chckpt[0])
            if 'mnli' in model_chckpt[1]: 
                config.num_labels = 3
            self.model = AutoModelForSequenceClassification.from_pretrained(model_chckpt[1], config=config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_chckpt)


    def get_inputs_embeds(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)


    @property
    def vocab(self):
        return self.model.get_input_embeddings().weight


    def forward(self, input_ids, attention_mask, token_type_ids, labels, inputs_embeds=None):  
        if inputs_embeds is not None:
            output = self.model(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=labels, inputs_embeds=inputs_embeds, return_dict=True)
        else:
            output = self.model(input_ids=input_ids,  attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=labels, return_dict=True)
        return output