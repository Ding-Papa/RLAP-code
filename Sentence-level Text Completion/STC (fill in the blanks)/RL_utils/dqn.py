import os, sys
from .replay_buffer import Memory
sys.path.append('../')
from model import ActorModel
import torch
import random
import torch.nn as nn
class MemoryBank:
    def __init__(self, buf_sz=10000):
        self.buffer = []
        self.buf_sz = buf_sz

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buf_sz:
            self.buffer = self.buffer[self.buf_sz//2:]
        self.buffer.append((state, action, reward, next_state, done))

class DQN:
    def __init__(self, plm, epsilon, tokenizer, gamma, buf_sz, batch_sz, lr, explore_update, mode='train'):
        self.batch_sz = batch_sz
        self.tokenizer = tokenizer
        self.memory = Memory(batch_size=batch_sz, max_size=buf_sz, beta=0.9)
        self.epsilon = epsilon
        self.gamma = gamma
        self.ucnt = 0
        self.explore_update = explore_update
        if mode == 'test':
            self.policy_net = ActorModel(plm).cuda()
        else:
            self.policy_net = ActorModel(plm).cuda()
            self.target_net = ActorModel(plm).cuda()
            for param in self.target_net.parameters():
                param.requires_grad = False
            self.target_net.load_state_dict(self.policy_net.state_dict(),strict=False)      
            self.opt = torch.optim.SGD(self.policy_net.parameters(), lr=lr)

    def select_action(self, cond, choices):
        if random.random() < self.epsilon:
            return torch.rand(len(choices))

        input_ids, attention_mask = [], []
        for choice in choices:
            output = self.tokenizer(cond, choice)
            input_ids.append(torch.IntTensor(output['input_ids'][:2048]))
            attention_mask.append(torch.IntTensor(output['attention_mask'][:2048]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                    padding_value=0,
                                    batch_first=True).cuda()
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            logits = self.policy_net(**inputs)

        if len(logits) > 1:
            return logits.squeeze()
        else:
            return logits[0]

    def select_next_action(self, cond, choices):
        if choices == []:
            return torch.tensor(0).cuda()
        input_ids, attention_mask = [], []
        for choice in choices:
            output = self.tokenizer(cond, choice)
            input_ids.append(torch.IntTensor(output['input_ids'][:2048]))
            attention_mask.append(torch.IntTensor(output['attention_mask'][:2048]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                    padding_value=0,
                                    batch_first=True).cuda()
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            logits = self.target_net(**inputs)
        return torch.max(logits.squeeze(), dim=0)[0]

    def getBatch(self):
        points, batch, importance_ratio = self.memory.get_mini_batches()
        state, action, reward, next_state, done = zip(*batch)

        new_cond, new_choices, new_blanks = zip(*next_state)
        input_ids = []
        attention_mask = []

        for (cond, choices, blanks), action in zip(state, action):
            output = self.tokenizer(cond, choices[action])
            input_ids.append(torch.IntTensor(output['input_ids'][:2048]))
            attention_mask.append(torch.IntTensor(output['attention_mask'][:2048]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                    padding_value=0,
                                    batch_first=True).cuda()

        reward = torch.tensor(reward).cuda()
        done = torch.tensor(done).cuda()
            
        return input_ids, attention_mask, \
                reward, \
                new_cond, new_choices, new_blanks, \
                done, points, importance_ratio

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
        
    def update(self):        
        self.ucnt += 1
        if self.ucnt % self.explore_update == 0 and self.epsilon > 0.02: 
            self.epsilon *= 0.95

        input_ids, attention_mask, \
        reward, \
        new_cond, new_choices, new_blanks, \
        done, points, importance_ratio = self.getBatch()
        
        next_q = torch.tensor([self.select_next_action(c, t) for (c, t) in zip(new_cond, new_choices)]).cuda()
        ground_truth = reward + self.gamma * next_q * (1 - done.long())
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        out = self.policy_net(**inputs).squeeze()

        td_error = torch.abs(out - ground_truth)

        loss = nn.MSELoss()(out,ground_truth)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.memory.update(points, td_error.cpu().detach().numpy())

    def save_weight(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weight(self, path):
        self.policy_net.load_state_dict(torch.load(path),strict=False)