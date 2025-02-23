# from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import copy
import random

class ACEDataset(Dataset):
    def __init__(self, data_path,data_type='rl', data_split=1):
        self.data_split = data_split
        self._load_schema()
        self._load_dataset(data_path)
        self._process_data()
        self.data_type = data_type
        self._gen_data_for_rl_model()

    def _load_schema(self):
        self.schema = {}
        with open('ACE05/ace05_event_schema.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                res = json.loads(line)
                self.schema[res['event_type']] = res['arguments']

    def _process_data(self):
        self.datas = []
        for data in self.label_datas:
            eve = {}
            for event in data['event_list']:
                if event['event_type'] not in eve.keys():
                    eve[event['event_type']] = []

                tmp_event = {
                    'event_type': event['event_type']
                }
                tmp_event['arguments'] = {}
                for argument in event['arguments']:
                    if argument['role'] not in tmp_event['arguments'].keys():
                        tmp_event['arguments'][argument['role']] = []
                    tmp_event['arguments'][argument['role']].append(argument['name'])

                event_list = [{}]
                for argument in tmp_event['arguments'].keys():
                    new_event_list = []
                    for el in event_list:
                        for entity in tmp_event['arguments'][argument]:
                            tmp_el = copy.deepcopy(el)
                            tmp_el[argument] = entity
                            new_event_list.append(tmp_el)
                    event_list = new_event_list
                eve[event['event_type']].extend(event_list)
            self.datas.append({
                    'text': data['text'],
                    'event_list': eve
                })

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            for event in data['event_list'].keys():
                new_data.append((data['text'], event, data['event_list'][event]))
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