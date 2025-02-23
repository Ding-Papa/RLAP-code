import torch
from dataset.process import cmrcDataset
import re
import copy

class ExtractionEnv:
    def __init__(self, llm_func, data_path, dataset='example', lang='zh', mode='train', data_split=0.8):
        self.data = None
        self.data_split = data_split
        self.llm_func = llm_func
        self.mode = mode
        self.dname = dataset
        self.index = 0
        self.lang = lang
        self.join = ''
        
        if dataset == 'cmrc2019':
            self.dataset = cmrcDataset(data_path=data_path, data_split=self.data_split)
            self.train_data = self.dataset.train_datas
            self.test_data = self.dataset.test_datas
            if self.mode == 'train':
                self.dataset_len = len(self.train_data)
            else:
                self.dataset_len = len(self.test_data)
    
    def choice_decision(self, cond, new_content, blanks):
        inst = '''
现在你是一个文字专家，需要完成一个句子填空的任务。我会给你一段不完整的文本段落【Incomplete Paragraph】，这个段落空缺的位置集合【BLANKS】，一个待填空的句子【Sentence】。
其中不完整文本段落【Incomplete Paragraph】中会存在某些空缺部分，用[BLANK]来表示，比如[BLANK1]代表第一个位置的空缺，[BLANK2]代表第二个位置的空缺，以此类推。
【BLANKS】是【Incomplete Paragraph】空缺位置的集合。你需要为待填空的句子【Sentence】在【Incomplete Paragraph】中选择一个合适填入的位置，并返回这个位置。比如你选择的是填充[BLANK1]，就返回[BLANK1]，你选择的是填充[BLANK2]，就返回[BLANK2]。
注意，你选择空缺必须存在于【BLANKS】中，比如【BLANKS】中没有[BLANK1]，则不能返回[BLANK1]。如果你认为没有正确选项，则随机返回一个位置。
切记，直接返回填充的位置，只返回一个答案，并且这个位置存在于【BLANKS】中，严禁有任何解释和多余的输出。
输入:'''
        inst += '【Incomplete Paragraph】: ' + cond + '\n【BLANKS】: ' + str(blanks) + '\n【Sentence】: ' + new_content + '\n输出:'
        position = self.llm_func(inst)
        if position == 'None':
            return self.choice_decision(cond, new_content, blanks)
        position = position.split('\n')[0].strip()
        position = position.strip("'")
        ans = [int(num) for num in re.findall(r'-?\d+', position)]
        if len(ans) != 1: 
            choose = torch.randint(0, len(blanks), (1,)).item()
            ans = [int(num) for num in re.findall(r'-?\d+', blanks[choose])]
        return ans[0]

    def step(self, cond, action, choices, blanks):
        new_content = choices[action]
        
        if len(choices) == len(blanks) == 1:
            if blanks[0][7] == ']':
                position = int(blanks[0][6])
            else:
                position = int(blanks[0][6:8])
            position_str = f'[BLANK{position}]'
            new_paragraph = cond.replace(blanks[0], new_content)
            done = True
        else:
            position = self.choice_decision(cond, new_content, blanks)
            position_str = f'[BLANK{position}]'
            if position_str not in blanks:
                position = torch.randint(0, len(blanks), (1,)).item()
                position_str = blanks[position]
            new_paragraph = cond.replace(position_str, new_content)
            done = False
        if self.sentence_to_choice[new_content] == self.answers[position-1]:
            score = 1
        else:
            score = 0

        new_choices = copy.deepcopy(choices)
        del new_choices[action]
        new_blanks = copy.deepcopy(blanks)
        new_blanks.remove(position_str)

        new_cond = f'{new_paragraph}'
        self.join = new_cond
        return (new_cond, new_choices, new_blanks), score, done

    def return_cond(self):
        return self.join

    def reset(self):
        if self.dname == 'cmrc2019':
            if self.mode == 'train':
                index = self.index % self.dataset_len
                self.index += 1
                self.data = self.train_data[index]
            else:
                index = self.index
                self.index += 1
                self.data = self.test_data[index]

            context = self.data[0]
            choices = self.data[1]
            self.answers = self.data[2]
            self.sentence_to_choice = self.data[3]
            blanks = self.data[5]
            
            return (context, choices, blanks), 0, False