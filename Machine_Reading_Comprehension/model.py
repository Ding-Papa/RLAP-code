import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math

class ActorModel(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.model = AutoModel.from_pretrained(plm_name,trust_remote_code=True)
        self.output_layer = nn.Linear(768, 1) 
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.output_layer(last_hidden_state[:,0,:])
        return logits

