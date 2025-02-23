from html import entities
import json
import torch
from dataset.squad import SQuADDataset
from dataset.cmrc import CMRCDataset
from dataset.C3mix import C3Dataset
from dataset.race import RACEDataset
import pdb
# from model import GlobalPointerModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.generation import GenerationConfig
import random, math,re
import copy
import numpy as np

# 该环境设定不包含中间步QA、动作空间也不含“停止阅读”
class ExtractionEnv:
    def __init__(self,llm_func,data_path, dataset='SQuAD2.0', lang='en', mode='train', max_data=1500, data_split=0.8):
        self.data = None
        self.state = None
        self.data_split = data_split
        self.llm_QA = llm_func
        self.mode = mode
        self.dname = dataset
        self.index = 0
        self.lang = lang
        self.max_data = max_data
        self.now_ans = ''

        if dataset == 'SQuAD2.0':
            self.dataset = SQuADDataset(data_path=data_path, mode=self.mode, max_data=self.max_data, data_split=self.data_split)
            self.dataset_len = len(self.dataset)
            self.qa_type='ext'
        elif dataset == 'CMRC18':
            self.dataset = CMRCDataset(data_path=data_path, mode=self.mode, max_data=self.max_data, data_split=self.data_split)
            self.dataset_len = len(self.dataset)
            self.qa_type='ext'
        elif dataset == 'C3mixed':
            self.dataset = C3Dataset(data_path=data_path, mode=self.mode, max_data=self.max_data, data_split=self.data_split)
            self.dataset_len = len(self.dataset)
            self.qa_type='choice'
        elif dataset == 'RACE':
            self.dataset = RACEDataset(data_path=data_path, mode=self.mode, max_data=self.max_data, data_split=self.data_split)
            self.dataset_len = len(self.dataset)
            self.qa_type='choice'
        
    def _example_generation(self, contexts, question):
        # 这里：中/英文QA的prompt
        if self.lang == 'zh' and self.qa_type == 'ext':
            inst='''
            现在进行抽取式问答任务，给定由几句话组成的文本片段、以及一个问题，请你从文本片段中抽取出合适的答案来回答给定的问题。
            注意：我提供的文本片段不一定是完整或者通畅的段落，在回答时你必须从给定的文本片段中提取出答案（也就是说，答案需要是给定文本的一个子串）、不能增删内容。
            设定一个问题只有一个回答，请提取最合适的答案。如果从文本片段中无法找到合适的回答，请返回“无法回答”。
            输入：'''
            inst+='文本片段：'+contexts+'\n问题：'+question+'\n回答：'
        elif self.lang == 'zh' and self.qa_type == 'choice':
            inst='''
            现在进行选择式问答任务，给定由几句话组成的文本片段、以及一个问题，请你从给定的候选答案中选择最正确、最合适的一个回答。
            注意：我提供的文本片段不一定是完整或者通畅的段落，你必须从给定的候选答案列表中选择一个作为回答、不能擅自增加或删除答案内容。
            设定一个问题只有一个回答，请选择并返回最合适的答案，不要有多余的输出。
            输入：'''
            inst+='文本片段：'+contexts+'\n问题：'+question+'\n回答：'
        elif self.lang == 'en' and self.qa_type == 'ext':
            inst = '''
            Now you are working on a reading comprehension task (Extractive QA). Given a piece of text, a question, please extract the appropriate answer from the text to answer the given question.
            Note: The text I provide may not be a complete paragraph, and you must extract the answer from the given text (i.e., the answer needs to be a substring of the given text), and you cannot add or delete words.
            There is only one answer to a question, please extract the most appropriate answer. if you cannot find the answer from the text, please return "no answer".
            Input:'''
            inst += 'Text: ' + contexts + '\nQuestion: ' + question + '\nAnswer:'
        elif self.lang == 'en' and self.qa_type == 'choice':
            inst = '''
            Now you are working on a reading comprehension task (multiple-choice QA). Given a piece of text and a question, please choose the most correct and appropriate answer from the given list of candidate answers.
            Note: The text I provide may not be a complete paragraph, you must choose an answer from the given list of candidate answers, and you cannot add or delete add or delete words.
            There is only one answer to a question, please select and return the most appropriate answer without redundant output.
            Input:'''
            inst += 'Text: ' + contexts + '\nQuestion: ' + question + '\nAnswer:'
        new_resp = self.llm_QA(inst)
        new_resp = new_resp.split('\n')[0].strip()
        new_resp = new_resp.strip("'")
        return new_resp.strip()

    def sigmoid(self, i):
        return 1 / (math.exp(-i) + 1)


    def choice_decision(self, ques, now_ans, next_context, context, sizeofaction):
        if self.lang=='en':
            new_context=(context + ' ' + next_context).strip()
        else:
            new_context=(context + next_context).strip()
        if sizeofaction!=0:
            new_ans = now_ans
            reward_value=0
        else:
            new_ans = self._example_generation(new_context, ques)
            reward_value = self.reward_test(new_ans, self.ground_truth)
        
        return new_ans, new_context, reward_value

    def reward_test(self, output, gold):
        for go in gold:
            if output==go:
                return 10
        return 0

    def step(self, ques, now_ans, context, action, choices):
        next_context = choices[action]
        new_ans, new_context, reward = self.choice_decision(ques, now_ans, next_context, context, len(choices)-1)
        self.now_ans = new_ans

        new_choices = copy.deepcopy(choices)
        del new_choices[action]
        
        if new_choices:
            done = False
        else:
            done = True
            print('问题+答案：',ques+'  '+new_ans)
        return (ques, new_ans, new_context, new_choices), reward, done

    def return_cond(self):
        return self.now_ans

    def reset(self):
        self.spo_list = {}
        if self.mode == 'train':
            index = self.index % self.dataset_len
            self.index += 1
        else:
            index = self.index
            self.index += 1

        self.data = self.dataset[index]
        if self.qa_type=='ext':
            ques = self.data[0]
            now_ans = self.data[1]
            self.ground_truth = self.data[-1]  
            choices = self.data[2]
        elif self.qa_type=='choice':
            if self.lang=='en':
                ques = self.data[0] + ' Candidate answers list: ' + str(self.data[1])
            elif self.lang=='zh':
                ques = self.data[0] + '候选答案列表：' + str(self.data[1])
            now_ans = self.data[2]
            self.ground_truth = [self.data[-1]]
            choices = self.data[3]

        return (ques, now_ans, '', choices), 0, False