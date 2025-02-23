from html import entities
import json
import torch
from dataset.nyt import NYTDataset
from dataset.duee import DuEEDataset
from dataset.hacred import HacREDDataset
from dataset.ske import SKEDataset
from dataset.ace05 import ACEDataset
import random, math,re
import copy
import numpy as np

class ExtractionEnv:
    def __init__(self, llm_ext_func,data_path, dataset='WebNLG', lang='en', mode='train', reward_type='v1', data_split=1):
        self.data = None
        self.state = None
        self.data_split = data_split
        self.llm_ext = llm_ext_func
        self.mode = mode
        self.dname = dataset
        self.index = 0
        self.lang = lang
        self.reward_type = reward_type

        if dataset == 'HacRED':
            self.dataset = HacREDDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'NYT':
            self.dataset = NYTDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'SKE':
            self.dataset = SKEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuEE1.0':
            self.dataset = DuEEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)
        elif dataset == 'ACE05':
            self.dataset = ACEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)
        

    def _load_schema(self):
        self.schema = {}
        if self.dname == 'DuEE1.0':
            with open('DuEE1.0/duee_event_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['event_type']] = [item['role'] for item in res['role_list']]
        if self.dname == 'ACE05':
            with open('ACE05/ace05_event_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['event_type']] = res['arguments']
    
    def _example_generation(self, input_texts, cond, hist_ext, slot_name):
        if self.dname == 'DuEE1.0':
            inst = '''现在在进行事件抽取任务，给定原句、事件类型、已经抽取出的内容，请你根据已有的内容按要求识别我想要的实体。
            我提供的输入会包含原句信息、事件类型、已经抽取的结果和我需要的元素，请你按照指定格式返回我想要的实体。原句中没有的信息不要考虑。
            要求输出内容仅包含我想要的实体，并且需要与输入中提供的事件类型和已抽取内容相匹配，不要有任何额外的输出。如果对应的实体有多个，请用英文逗号分隔，如果没有相应的实体，返回'None'。
            输入：'''
            dic = {'原句':input_texts,'事件类型':cond,'已经抽取的内容':hist_ext,'我想要的元素':slot_name}
            inst = inst + str(dic) + '\n' + '输出：'
        elif self.dname == 'ACE05':
            inst = '''I am now working on an event extraction task, given the original sentence, event type, and what has been extracted, I am asking you to identify the elements I want according to the given content.
            The input I provide will contain the information of the original sentence, the event type, the already extracted results and the elements I need, please return the entities I want in the specified format. Do not consider information that is not in the original sentence.
            The output is required to contain only the entities I want and needs to match the event types and already extracted content provided in the input, without any additional output. If there is more than one corresponding entity, please separate them with English commas, if there is no corresponding entity, return 'None'.
            INPUT:'''
            dic = {'text':input_texts,'event_type':cond,'extracted content':hist_ext,'elements I want':slot_name}
            inst = inst + str(dic) + '\n' + '输出：'
        elif self.lang == 'zh':
            inst = '''现在在进行关系三元组抽取任务，给定原句、感兴趣的关系类型，也可能会给定已经抽取出的内容，请你根据已有的内容按要求识别我想要的实体。
            我提供的输入会包含原句信息、关系类型、已经抽取的结果和我需要的实体类型，请你按照指定格式返回我想要的实体。关系类型字段如果有括号“（）”，括号内表示的是主语或宾语应当符合的类别要求。
            要求输出内容仅包含我想要的实体，并且需要与输入中提供的关系类型和已抽取内容相匹配，不要有任何额外的输出。如果对应的实体有多个，请用英文逗号分隔，如果没有相应的实体，返回'None'。
            输入：'''
            dic = {'原句':input_texts,'关系类型':cond,'已经抽取的内容':hist_ext,'我想要的实体类型':slot_name}
            inst = inst + str(dic) + '\n' + '输出：'
        elif self.lang == 'en':
            inst = '''Now we are doing the task of relation triple extraction. Given the text and the interested relation, we may also give the extracted content. Please identify the entities I want according to the given content.
            The input I provide will contain the text information, the relation, the extracted results and the entity type I need. Please return the entities I want in the specified format. If there are parentheses in the 'Relation', the contents of the parentheses indicate the requirements of the category to be met by the subject or object.
            The output content only contains the entities I want, and needs to match the relation and extracted content provided in the input. If there are multiple corresponding entities, please separate them with English commas. If there is no corresponding entity, return 'None'.
            INPUT:'''
            dic = {'text':input_texts,'relation':cond,'extracted content':hist_ext,'entity type I need':slot_name}
            inst = inst + str(dic) + '\n' + 'OUTPUT:'
        new_resp = self.llm_ext(inst)
        new_resp = new_resp.split('\n')[0].strip()
        new_resp = new_resp.strip("'")
        return new_resp.strip()

    def sigmoid(self, i):
        return 1 / (math.exp(-i) + 1)

    def score2prob(self, entities):
        entities_mention = list(set([e[0] for e in entities]))
        logsum = sum([math.exp(e[1]) for e in entities])
        entities = [(e[0],math.exp(e[1])/logsum, e[1]) for e in entities]
        entities_score = [(name, sum([i[1] for i in entities if i[0] == name]), max([i[2] for i in entities if i[0] == name])) for name in entities_mention]
        return entities_score

    def choice_decision(self, cond, choices, action):
        """ print('='* 50)
        print(choices[action]) """
        cond_list = cond.split(';')
        rel_type = cond_list[0].strip()
        hist = {}
        try:
            for item in cond_list[1:]:
                kv = item.split(':')
                hist[kv[0].strip()] = kv[1].strip()
        except:
            hist = {}
        ori_resp = self._example_generation(self.text, rel_type, hist, choices[action])
        std_ans = ''
        for spo in self.spo_list[cond]:
            if self.dname == 'DuEE1.0':
                if spo[choices[action]] == '[None]':
                    std_ans += ''
                else:
                    std_ans += ',' + spo[choices[action]].strip()
            else:
                if spo[choices[action]] == '[None]':
                    std_ans += ''
                else:
                    std_ans += ','+spo[choices[action]].strip()
        std_ans = std_ans.strip(',')
        std_ans = list(set(std_ans.split(',')))
        std_ans = ','.join(std_ans)
        entities_1step = self._recognize(ori_resp, threshold=0.39, std_ans=std_ans)
        entities_1step = self.score2prob(entities_1step)
        if self.dname == 'DuEE1.0' or 'ACE05':
            if entities_1step == []:
                if std_ans.strip():
                    entities_1step.append(('[None]',0.9, 0.0))
                else:
                    entities_1step.append(('[None]',0.9, 9.99))
        elif entities_1step == []:
            entities_1step.append(('[None]',0.9, 0.0)) 
        return entities_1step

    def reward_assign(self, output, gold):
        output=set(output.split(','))
        gold=set(gold.split(','))
        inter = output & gold
        return len(inter)/len(gold)

    def _recognize(self, llm_resp, threshold = 0.8, std_ans=''):
        if llm_resp == 'None':
            return []
        entity_list = llm_resp.split(',')
        entities = []
        std_ans = std_ans.strip()
        if not std_ans:
            entities = [(en.strip(), 0.0) for en in entity_list]
            return entities
        score=self.reward_assign(llm_resp,std_ans)
        if score>threshold:
            for ent in entity_list:
                entities.append((ent.strip(), score*10))
        if entities:
            entities.sort(key=lambda x:x[-1], reverse=True)
        return entities

    def step(self, cond, action, choices):
        slot_name = choices[action]
        entities = self.choice_decision(cond, choices, action)
        all_score = sum([entity[2] for entity in entities]) / len(entities)

        entities = list(set([e[0] for e in entities]))
        valid_conds = []
        for entity in entities:
            new_cond = f'{cond}; {slot_name}:{entity}'
            if new_cond not in self.spo_list.keys():
                self.spo_list[new_cond] = []
            valid_conds.append(new_cond)
            for spo in self.spo_list[cond]:
                if all_score>6:
                    self.spo_list[new_cond].append(spo)

        new_choices = copy.deepcopy(choices)
        del new_choices[action]

        if new_choices:
            done = False
        else:
            done = True
        return [(_cond , self.text, new_choices) for _cond in valid_conds], all_score, done


    def return_cond(self):
        return self.spo_list

    def slot_fill_(self, slot_list, cond):
        if self.dname == 'DuEE1.0' or 'ACE05':
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        rel[slot_name] = '[None]'
        else:
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        rel[slot_name] = '[None]'

    def reset_with_input(self, text, cond, choices):
        self.spo_list = {}
        self.spo_list[cond] = {}
        self.text = text
        self.gt_num = 1e12
        return [(cond, text, choices)], 0, False

    def reset(self):
        self.spo_list = {}
        if self.mode == 'train':
            index = self.index % self.dataset_len
            self.index += 1
        else:
            index = self.index
            self.index += 1

        self.data = self.dataset[index]
        cond = self.data[1]
        text = self.data[0]
        self.gt_num = len(self.data[2])
        self.spo_list[cond] = self.data[2]
        if self.dname in ['NYT']:
            choices = ['subject', 'object']
        elif self.dname in ['HacRED','SKE']:
            choices = ['主语','宾语']
        elif self.dname == 'DuEE1.0' or 'ACE05':
            choices = self.schema[cond]
            self.slot_fill_(choices, cond)
        self.text = text
        return [(cond, self.text, choices)], 0, False