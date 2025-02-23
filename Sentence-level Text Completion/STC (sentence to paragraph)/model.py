import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActorModel(nn.Module):
    def __init__(self, plm):
        super(ActorModel, self).__init__()
        self.plm = AutoModelForCausalLM.from_pretrained(plm, trust_remote_code=True)
        if plm == 'your model path of Qwen2.5-7B-Instruct':
            self.output_layer = nn.Linear(3584, 1)
        elif plm == 'your model path of Mistral-7B-Instruct-v0.3':
            self.output_layer = nn.Linear(4096, 1)
        elif plm == 'your model path of Llama-3-8B-Instruct':
            self.output_layer = nn.Linear(4096, 1)
    
    def get_eos_embedding(self, inputs):
        embeddings = []
        outputs = self.plm(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        eos_token_id = self.plm.config.eos_token_id
        for i, input_id in enumerate(inputs['input_ids']):
            eos_position = (input_id == eos_token_id).nonzero(as_tuple=True)[0]
            eos_embedding = last_hidden_state[i, eos_position, :]
            eos_embedding = eos_embedding / 10
            embeddings.append(eos_embedding)
        return torch.stack(embeddings)
    
    def forward(self, inputs):
        with torch.no_grad():
            eos_embeddings = self.get_eos_embedding(inputs)
        logits = self.output_layer(eos_embeddings)
        return logits.squeeze(-1)