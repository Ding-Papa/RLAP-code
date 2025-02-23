from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import copy
import random

class HacREDDataset(Dataset):
    def __init__(self, data_path, data_type='rl', data_split=1):
        self.data_split = data_split
        self._load_dataset(data_path)
        self._process_data()
        self.data_type = data_type
        self._gen_data_for_rl_model()

    def _process_data(self):
        self.datas = []
        for data in self.label_datas:
            rel = {}
            for relation in data['relationMentions']:
                if relation['label'] not in rel.keys():
                    rel[relation['label']] = []
                new_relation = {
                    '主语': relation['em1Text'],
                    '关系': relation['label'],
                    '宾语': relation['em2Text']
                }
                rel[relation['label']].append(new_relation)
            self.datas.append({
                'text': data['sentText'],
                'relation_list': rel  
            })

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

        dataset_len = len(self.label_datas)
        if 'train' in data_path:
            if self.data_split == 1:
                self.label_datas = self.label_datas[:dataset_len // 2]
            else:
                self.label_datas = self.label_datas[dataset_len // 2:]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        examples = self.datas[index]
        return examples