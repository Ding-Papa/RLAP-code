import torch
import json
from tqdm import tqdm
import argparse
import random
from transformers import BertTokenizerFast
from Environment import ExtractionEnv
from RL_utils.dqn import DQN
from collections import Counter
from LLM_server import qwen25_14B_api, mistral_7B_api, qwen_14B_api


parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=777, help="seed")
parser.add_argument("--lang", type=str, default='zh', choices=['zh','en'], help='language for predict')
parser.add_argument("--dataset", type=str, choices=['wiki80','WebNLG','DuIE2.0','HacRED','NYT10','SKE','NYT11-HRL'])
parser.add_argument('--action_mode', type=str, default='RL', choices=['RL','random','sequence'])
parser.add_argument("--rl_model", type=str, default='rl_HacRED_gpt', help='choice of rl model file')
parser.add_argument("--write_dir", type=str, default='../predicted/', help='directory to save results')
parser.add_argument("--llm_extractor", type=str, default='chatgpt', choices=['chatgpt','llama3_8B','qwen_14B','mistral_7B','mixtral_87','qwen25_14B'],help='choice of llm_extractor')
args = parser.parse_args()

#Params
dataset = args.dataset
lang = args.lang
write_dir = args.write_dir
if args.action_mode == 'RL':
    model_name = args.rl_model
else:
    model_name = args.action_mode
    plm = 'your_actormodel_path'
llm_name = args.llm_extractor
llm_ext_func = eval(llm_name + '_api')
random.seed(777)

with open(f'../{dataset}/new_test.json', 'r', encoding='utf-8') as f:
    datas = []
    for line in f.readlines():
        datas.append(json.loads(line))

tokenizer = BertTokenizerFast.from_pretrained(plm)
if model_name == 'random' or model_name == 'sequence':
    agent1 = None
else:
    agent1 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
    agent1.load_weight(f'weight/{model_name}.pt')

env = ExtractionEnv(llm_ext_func=llm_ext_func,
                data_path=f'../{dataset}/new_test.json',
                dataset=dataset,
                mode='test',
                lang=lang)

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)


    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return (prec, reca, f1)

def ext_with_env(text, ori_cond, choices, model_name):
    state_list, _, _ = env.reset_with_input(text, ori_cond, choices)
    slot_list = state_list[0][2]
    slot_num = len(slot_list)
    ep_reward = 0
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, text, choices = state
            if model_name == 'sequence':
                action = 0
            elif model_name == 'random':
                action = random.randint(0, len(choices) - 1)
            else:
                action = agent1.select_action(cond, text, choices)
                action = torch.argmax(action)
            next_state_list, reward, done = env.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break
    pre_list = []
    predict_list = env.return_cond()
    for k in predict_list.keys():
        if '[None]' in k:
            continue
        c = Counter(k)
        if c[';'] == 0:
            gt = predict_list[k]   # ground truth
        if c[';'] == slot_num:
            pre = {'relation': ori_cond}
            predict_word_offset = []
            for slot in slot_list:
                predict_word_offset.append((k.index('; ' + slot+':'), len(slot) + 1, slot))
            predict_word_offset.sort()
            for index, offset in enumerate(predict_word_offset):
                s, l, slot = offset
                vs = s + 2 + l
                if index != len(slot_list) - 1:
                    ve = predict_word_offset[index+1][0]
                else:
                    ve = len(k)
                pre[slot] = k[vs:ve]
            pre_list.append(pre)
    return pre_list

def spo2text_zh(spo): return spo['relation'] + '|' + spo['主语'] + '|' + spo['宾语']
def spo2text_en(spo): return spo['relation'] + '|' + spo['subject'] + '|' + spo['object']
def spo2text_gt(spo): return spo['label'] + '|' + spo['em1Text'] + '|' + spo['em2Text']

f1 = MetricF1()
wlist = []
ind = 0
for data in tqdm(datas):
    ind += 1
    wdic = {}
    wdic['text'] = data['sentText']
    wdic['std_ans'] = data['relationMentions']
    all_relations = [q['label'] for q in data['relationMentions']]
    all_relations = list(set(all_relations))
    wdic['preds'] = []
    pred = set()
    for rel_cond in all_relations:
        if lang == 'zh':
            predict = ext_with_env(data['sentText'], rel_cond, ['主语','宾语'], model_name)
        else:
            predict = ext_with_env(data['sentText'], rel_cond, ['subject','object'], model_name)
        for spo in predict:
            if lang == 'zh':
                pred.add(spo2text_zh(spo))
            else:
                pred.add(spo2text_en(spo))
        wdic['preds'] += predict
    print(wdic)
    wlist.append(wdic)
    gold = set([spo2text_gt(spo) for spo in data['relationMentions']])
    f1.append_relax(pred, gold)
    if ind % 50 == 0:
        wrt=f1.compute()
        with open(f'{write_dir}{dataset}/{llm_name}_{model_name}.json', 'a', encoding='utf-8') as f:
            json.dump(wlist,f,ensure_ascii=False,indent=4)
            f.write('\n'+str(wrt)+'\n')
        wlist = []

f1.compute()