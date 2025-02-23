# from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import copy
import random

class CMRCDataset(Dataset):
    def __init__(self, data_path, mode='train', max_data=2000, data_split=0.8):
        self.data_split = data_split
        self.data_mode = mode
        self._load_dataset(data_path,max_data)
        self._gen_data_for_rl_model()

    def _gen_data_for_rl_model(self):
        self.datas = []
        for data in tqdm(self.label_datas, desc='Process data for rl model'):
            context_list = data['context'].split('。')
            context_list = [context.strip() + '。' for context in context_list if context.strip() != '']
            context_list = [context for context in context_list if len(context) > 3]
            if len(context_list)<6: continue
            for qap in data['Q_Apairs'][:3]:
                if not qap["answers"]: qap["answers"]=["无法回答"]
                self.datas.append((qap["question"], "", context_list, qap["answers"]))
        

    def _load_dataset(self, data_path,maxdata):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.label_datas = json.load(f)

        dataset_len = len(self.label_datas)
        maxdata = min(maxdata,dataset_len)
        if self.data_mode == 'train':
            self.label_datas= self.label_datas[:int(maxdata * self.data_split)]
            self.valid_data = self.label_datas[int(maxdata * self.data_split):]
        else:
            self.label_datas = self.label_datas[:maxdata]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        examples = self.datas[index]
        return examples