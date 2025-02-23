from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import copy
import random

class NYTDataset(Dataset):
    def __init__(self, data_path, data_type='rl', data_split=1):
        self.data_split = data_split
        self._load_dataset(data_path)
        self.data_type = data_type
        self._process_data()
        self._gen_data_for_rl_model()

    def _process_data(self):
        self.datas = []
        for data in self.label_datas:
            rel = {}
            for relation in data['relationMentions']:
                if relation['label'] not in rel.keys():
                    rel[relation['label']] = []
                new_relation = {
                    'subject': relation['em1Text'],
                    'relation': relation['label'],
                    'object': relation['em2Text']
                }
                rel[relation['label']].append(new_relation)
            self.datas.append({
                'text': data['sentText'],
                'relation_list': rel
            })

    def _load_rel_map(self, path):
        with open(path.replace('new_train.json','rel2id.json'), 'r', encoding='utf-8') as f:
            self.rel_map = json.load(f)

    def _gen_data_for_classification_model(self):
        self.datas = []
        for data in self.label_datas:
            output = self.tokenizer(data['sentText'], return_token_type_ids=True, return_offsets_mapping=True)
            input_ids = output['input_ids'][:512]
            token_type_ids = output['token_type_ids'][:512]
            labels = torch.zeros(len(self.rel_map)).int()
            for relation in data['relationMentions']:
                labels[self.rel_map[relation['label']]] = 1
            self.datas.append((torch.IntTensor(input_ids), torch.IntTensor(token_type_ids), labels))

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            for relation in data['relation_list'].keys():
                if len(data['relation_list'][relation]) >= 0:
                    new_data.append((data['text'], relation, data['relation_list'][relation]))
        self.datas = new_data

    def _load_dataset(self, data_path):
        self.label_datas = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.label_datas.append(json.loads(line))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        examples = self.datas[index]
        return examples