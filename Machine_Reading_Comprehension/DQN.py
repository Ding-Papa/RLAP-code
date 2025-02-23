from model import ActorModel
import torch
import torch.nn as nn
import random
from transformers import AutoTokenizer, set_seed
from Environment import ExtractionEnv
import argparse
import numpy as np
from tqdm import tqdm
from RL_utils.dqn import DQN
import math
from LLM_server import mistral_7B_api, qwen_14B_api, qwen25_14B_api
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=888, help="seed")
parser.add_argument("--plm", type=str, default='model/gte-multilingual-base', help='your path for foundation model (actor model)')
parser.add_argument("--exploration_update", type=int, default=100, help='how many step to update exploration ratio')
parser.add_argument("--max_step", type=int, default=15, help='max step on a single episode')
parser.add_argument("--num_episode", type=int, default=6, help='sample episode number')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate for the DQN model')
parser.add_argument("--do_train", type=bool, default=False, help='whether train')
parser.add_argument("--max_data", type=int, default=1500, help='use how many data to train DQN')
parser.add_argument("--do_test", type=bool, default=False, help='whether test')
parser.add_argument("--weight_file", type=str, default='rl_SQuAD_mistral.pt', help='path to save rl model weight')
parser.add_argument('--dname', type=str, choices=['SQuAD2.0','CMRC18','C3mixed','RACE'])
parser.add_argument('--buf_size', type=int, default=4000, help='the size of the memory size')
parser.add_argument('--action_strategy', type=str, default='RL', choices=['RL','Random','Sequence'])
parser.add_argument('--data_split', type=float, default=1)
args = parser.parse_args()

"""
Parameters
"""
seed = args.seed
plm = args.plm
lr = args.learning_rate
target_update = 20
iters_save = 500

if args.do_train:
    """
    Training
    """
    
    np.random.seed(seed)
    set_seed(seed)
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_episode,
        "batch_size": 32,
        "seed": args.seed,
    }
    tokenizer = AutoTokenizer.from_pretrained(plm, trust_remote_code=True)
    if args.dname == 'CMRC18':
        env = ExtractionEnv(llm_func=qwen_14B_api,
                data_path='CMRC2018/new_train.json',
                dataset='CMRC18',
                mode='train',
                lang='zh',
                max_data=args.max_data,
                data_split=args.data_split)
    elif args.dname == 'SQuAD2.0':
        env = ExtractionEnv(llm_func=mistral_7B_api,
                data_path='SQuAD2.0/new_train.json',
                dataset='SQuAD2.0',
                mode='train',
                lang='en',
                max_data=args.max_data,
                data_split=args.data_split)
    elif args.dname == 'C3mixed':
        env = ExtractionEnv(llm_func=qwen_14B_api,
                data_path='C3_mixed/new_train.json',
                dataset='C3mixed',
                mode='train',
                lang='zh',
                max_data=args.max_data,
                data_split=args.data_split)        
    elif args.dname == 'RACE':
        env = ExtractionEnv(llm_func=mistral_7B_api,
                data_path='RACE/new_train_high.json',
                dataset='RACE',
                mode='train',
                lang='en',
                max_data=args.max_data,
                data_split=args.data_split)    
        
    tot_step = args.num_episode * env.dataset_len
    print("dataset_len:", env.dataset_len)
    n = math.log(0.05 / 0.9) / math.log(0.95)
    explore_update = int(tot_step // n)
    print('Explore update:', explore_update)
    agent = DQN(plm=plm,epsilon=0.9, tokenizer=tokenizer, gamma=0.5,buf_sz=args.buf_size,batch_sz=32, lr=lr, explore_update=explore_update,mode='train')
    
    print('Begin RL training!')

    rewards = []
    moving_average_rewards = []
    ep_steps = []
    for i_episode in tqdm(range(tot_step), desc='Training RL agent'):
        state, _, _ = env.reset()
        ep_reward = 0
        for i_step in range(args.max_step): 
            ques, now_ans, context, choices = state
            action = agent.select_action(ques, now_ans, context, choices)
            action = torch.argmax(action)
            next_state, reward, done = env.step(ques,now_ans,context,action,choices)
            ep_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            print('reward:'+str(reward))
            state= next_state
            if done:
                break
        agent.update() # 每步更新网络
        print('ep_reward:'+str(ep_reward))
        if i_episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if i_episode % iters_save == 0 and i_episode != 0:
            print('save at the step:', i_episode)
            print('************************************************')
            agent.save_weight(f'weight/{args.weight_file}')

        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1] + 0.1*ep_reward)

if args.do_test:
    tokenizer = AutoTokenizer.from_pretrained(plm, trust_remote_code=True)
    if args.dname == 'CMRC18':
        env = ExtractionEnv(llm_func=qwen25_14B_api,
                data_path='CMRC2018/new_train.json',
                dataset='CMRC18',
                mode='test',
                lang='zh',
                max_data=args.max_data,
                data_split=args.data_split)
    elif args.dname == 'SQuAD2.0':
        env = ExtractionEnv(llm_func=qwen25_14B_api,
                data_path='SQuAD2.0/new_train.json',
                dataset='SQuAD2.0',
                mode='test',
                lang='en',
                max_data=args.max_data,
                data_split=args.data_split)
    elif args.dname == 'C3mixed':
        env = ExtractionEnv(llm_func=qwen25_14B_api,
                data_path='C3_mixed/new_train.json',
                dataset='C3mixed',
                mode='test',
                lang='zh',
                max_data=args.max_data,
                data_split=args.data_split)        
    elif args.dname == 'RACE':
        env = ExtractionEnv(llm_func=qwen25_14B_api,
                data_path='RACE/new_train_high.json',
                dataset='RACE',
                mode='test',
                lang='en',
                max_data=args.max_data,
                data_split=args.data_split)   

    if args.action_strategy == 'RL':
        agent = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=args.buf_size,batch_sz=32, lr=lr, explore_update = 1e10, mode='test')
        agent.load_weight(f'weight/{args.weight_file}')

    print('Begin RL testing!')
    total_acc=0
    total_cnt=0
    wlist=[]

    for i_episode in tqdm(range(env.dataset_len),desc='Extract through RL agent'):
        state, _, _ = env.reset()
        ep_reward = 0
        for i_step in range(args.max_step):
            ques, now_ans, context, choices = state
            if args.action_strategy == 'RL':
                action = agent.select_action(ques, now_ans, context, choices)
                action = torch.argmax(action)
            # Random
            elif args.action_strategy == 'Random':
                action = random.randint(0, len(choices) - 1)
            # Seq
            elif args.action_strategy == 'Sequence':
                action = 0
            next_state, reward, done = env.step(ques,now_ans,context,action,choices)
            state = next_state
            if done:
                break
            
        pred_ans=env.return_cond()
        if env.reward_test(pred_ans,env.ground_truth,threshold=0.9):
            total_acc+=1
        total_cnt+=1
        wlist.append({'ques':state[0],'pred_ans':pred_ans,'gt':env.ground_truth})
        
    print('Test acc:',total_acc/total_cnt)
    with open('./predict/'+args.dname+'_'+args.action_strategy+'_qwen25_14B'+'.json','w') as f:
        json.dump(wlist,f,ensure_ascii=False,indent=4)
