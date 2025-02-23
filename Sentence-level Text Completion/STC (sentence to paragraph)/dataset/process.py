import json
from tqdm import tqdm
from torch.utils.data import Dataset

class HacREDDataset(Dataset):
    def __init__(self, data_path, data_split):
        self.data_split = data_split
        self._load_dataset(data_path)
        self._gen_data_for_rl_model()

    def _load_dataset(self, data_path):
        self.label_datas = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.label_datas.append(json.loads(line))

        dataset_len = len(self.label_datas)
        self.train_data = self.label_datas[:int(dataset_len * self.data_split)]
        self.test_data = self.label_datas[int(dataset_len * self.data_split):]

    def _gen_data_for_rl_model(self):
        self.train_datas = []
        for data in tqdm(self.train_data, desc='Process data for rl model'):
            self.train_datas.append((data['sentence_list'], data['shuffled_sentences'], data['paragraph'], data['indices'], data['mapping']))

        self.test_datas = []
        for data in tqdm(self.test_data, desc='Process data for rl model'):
            self.test_datas.append((data['sentence_list'], data['shuffled_sentences'], data['paragraph'], data['indices'], data['mapping']))
        
        
        
class SQuADDataset(Dataset):
    def __init__(self, data_path, data_split):
        self.data_split = data_split
        self._load_dataset(data_path)
        self._gen_data_for_rl_model()

    def _load_dataset(self, data_path):
        self.label_datas = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.label_datas.append(json.loads(line))

        dataset_len = len(self.label_datas)
        self.train_data = self.label_datas[:2000]
        self.test_data = self.label_datas[2000: 2500]

    def _gen_data_for_rl_model(self):
        self.train_datas = []
        for data in tqdm(self.train_data, desc='Process data for rl model'):
            self.train_datas.append((data['sentence_list'], data['shuffled_sentences'], data['paragraph'], data['indices'], data['mapping']))

        self.test_datas = []
        for data in tqdm(self.test_data, desc='Process data for rl model'):
            self.test_datas.append((data['sentence_list'], data['shuffled_sentences'], data['paragraph'], data['indices'], data['mapping']))