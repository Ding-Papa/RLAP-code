import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ActorModel(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.output_layer = nn.Linear(768, 1)
    
    def forward(self, input_ids, token_type_ids):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = self.output_layer(last_hidden_state[:,0,:])
        return logits