from torch import nn

from .utils import unfreeze_params


class BaseBertVictim(nn.Module):
    def __init__(self, model, layer):
        super(BaseBertVictim, self).__init__()
        self.model = model
        self.layer = layer

    def preprocess_model(self):
        if hasattr(self.model, "pooler"):
            self.model.pooler = None

        unfreeze_params(self.model.encoder)
        self.model.eval()

    @property
    def vocab_size(self):
        return self.vocab().shape[0]

    @property
    def vocab(self):
        return self.model.embeddings.word_embeddings.weight

    def get_inputs_embeds(self, input_ids):
        return self.model.embeddings.word_embeddings(input_ids)

    def forward(self, inputs_embeds, attention_mask, token_type_ids, **kwargs):
        output = self.model(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        return output.last_hidden_state * attention_mask.unsqueeze(-1)


class BertVictim(BaseBertVictim):
    def __init__(self, model, layer=0):
        super(BertVictim, self).__init__(model, layer)
        self.model.encoder.layer = self.model.encoder.layer[: self.layer + 1]
        self.preprocess_model()


class AlbertVictim(BaseBertVictim):
    def __init__(self, model, layer=0):
        super().__init__(model, layer)
        self.preprocess_model()

    def preprocess_model(self):
        group_idx = int(self.layer /
                        (self.model.config.num_hidden_layers
                        / self.model.config.num_hidden_groups))
        self.model.encoder.albert_layer_groups = self.model.encoder.albert_layer_groups[:group_idx + 1]
        self.model.config.num_hidden_layers = self.layer + 1
        super().preprocess_model()
