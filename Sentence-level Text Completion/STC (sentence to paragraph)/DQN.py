import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from Environment import ExtractionEnv
import argparse
import numpy as np
from tqdm import tqdm
from RL_utils.dqn import DQN
import math
import json
import random
from llm_server import qwen25_14B_api, llama3_8B_api

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=888, help="seed")
parser.add_argument("--exploration_update", type=int, default=100, help='how many step to update exploration ratio')
parser.add_argument("--max_step", type=int, default=15, help='max step on a single episode')
parser.add_argument("--num_episode", type=int, default=10, help='sample episode number')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate for the DQN model')
parser.add_argument("--do_train", type=bool, default=False, help='whether train')
parser.add_argument("--do_test", type=bool, default=False, help='whether test')
parser.add_argument("--test_mode", type=str, choices=['rl', 'random', 'prompt_no_icl', 'prompt_icl'])
parser.add_argument("--weight_file", type=str, default='rl_NYT_gpt.pt', help='path to save rl model weight')
parser.add_argument('--task_type', type=str, choices=['sentences_to_paragraph_zh', 'sentences_to_paragraph_en'])
parser.add_argument('--buf_size', type=int, default=5000, help='the size of the memory size')
parser.add_argument('--data_split', type=float, default=0.7)
args = parser.parse_args()

"""
Parameters
"""
seed = args.seed
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
  
    if args.task_type == 'sentences_to_paragraph_zh':
        env = ExtractionEnv(llm_func=None,
                            data_path='your data path of HacRED.json', 
                            dataset='HacRED',
                            lang='zh',
                            mode='train',
                            data_split=args.data_split)
        plm = 'your model path of Qwen2.5-7B-Instruct'
    elif args.task_type == 'sentences_to_paragraph_en':
        env = ExtractionEnv(llm_func=None,
                            data_path='your data path of SQuAD2.0.json', 
                            dataset='SQuAD2.0',
                            lang='en',
                            mode='train',
                            data_split=args.data_split)
        plm = "your model path of Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(plm, trust_remote_code=True)
    
    tot_step = args.num_episode * env.dataset_len
    n = math.log(0.05 / 0.9) / math.log(0.95)
    explore_update = int(tot_step // n)
    print('Explore update:', explore_update)
    agent = DQN(plm=plm,
                epsilon=0.9,
                tokenizer=tokenizer, 
                gamma=0.5,
                buf_sz=args.buf_size,
                batch_sz=32, 
                lr=lr, 
                explore_update=explore_update,
                mode='train')

    print('Begin RL training!')

    rewards = []
    moving_average_rewards = []
    ep_steps = []
    for i_episode in tqdm(range(tot_step), desc='Training RL agent'):
        state, _, _ = env.reset()
        ep_reward = 0
        rl_indices = []
        for i_step in range(args.max_step):
            cond, choices = state 
            action = agent.select_action(cond, choices)
            action = torch.argmax(action)
            new_index = env.mapping[choices[action]]
            rl_indices.append(new_index)
            next_state, reward, done = env.step(cond, action, choices, rl_indices)
            ep_reward += reward

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.update()

        if i_episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if i_episode % iters_save == 0 and i_episode != 0:
            agent.save_weight(f'weight/{args.weight_file}')

        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1] + 0.1*ep_reward)



if args.do_test:
    def count_consistent_order_optimized(env_target, rl_indices):
        target_map = {value: index for index, value in enumerate(env_target)}
        rl_map = {value: index for index, value in enumerate(rl_indices)}
            
        count = 0
        n = len(env_target)
        
        for i in range(n):
            for j in range(i + 1, n):
                target_order = target_map[i] < target_map[j]
                rl_order = rl_map[i] < rl_map[j]
                if target_order == rl_order:
                    count += 1
                    
        return count
        
    if args.test_mode == 'rl':
        if args.task_type == 'sentences_to_paragraph_zh':
            env = ExtractionEnv(llm_func=None,
                                data_path='your data path of HacRED.json', 
                                dataset='HacRED',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
            plm = 'your model path of Qwen2.5-7B-Instruct'
        elif args.task_type == 'sentences_to_paragraph_en':
            env = ExtractionEnv(llm_func=None,
                                data_path='your data path of SQuAD2.0.json', 
                                dataset='SQuAD2.0',
                                lang='en',
                                mode='test',
                                data_split=args.data_split)
            plm = "your model path of Llama-3-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(plm, trust_remote_code=True)
            
        agent = DQN(plm=plm,
                    epsilon=0, 
                    tokenizer=tokenizer, 
                    gamma=0.5,
                    buf_sz=10000,
                    batch_sz=32, 
                    lr=lr, 
                    explore_update = 1e10,
                    mode='test')
        agent.load_weight(f'weight/{args.weight_file}')

        print('Begin RL testing!')
        total_score = 0
        relative_position_score = 0

        for i_episode in tqdm(range(env.dataset_len), desc='Sentences to Paragraph through RL agent'):
            state, _, _ = env.reset()
            sentences = env.sentence_list
            paragraph = env.paragraph
            rl_indices = []
            for i_step in range(args.max_step):
                cond, choices = state
                action = agent.select_action(cond, choices)
                action = torch.argmax(action)
                rl_indices.append(env.mapping[choices[action]])
                next_state, reward, done = env.step(cond, action, choices, rl_indices)
                state = next_state
                if done:
                    break
            
            predict = env.return_cond()
            score = 1 if rl_indices == env.target else 0
            total_score += score
            relative_position = count_consistent_order_optimized(env.target, rl_indices) / ((len(env.target)*(len(env.target)-1))//2)
            relative_position_score += relative_position
            
            if args.task_type == 'sentences_to_paragraph_zh':
                save_predict_path = "your prediction save path"
            elif args.task_type == 'sentences_to_paragraph_en':
                save_predict_path = "your prediction save path"
                
            with open(save_predict_path, 'a') as f:
                f.write(json.dumps({'sentences': sentences, 'paragraph': paragraph, 'predict': predict, 'score': score}, ensure_ascii=False) + '\n')
        
        print('CAC: ', total_score / env.dataset_len) 
        print('SOC: ', relative_position_score / env.dataset_len)
        
    elif args.test_mode == 'random':
        if args.task_type == 'sentences_to_paragraph_zh':
            env = ExtractionEnv(llm_func=None,
                                data_path='your data path of HacRED.json', 
                                dataset='HacRED',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
            plm = 'your model path of Qwen2.5-7B-Instruct'
        elif args.task_type == 'sentences_to_paragraph_en':
            env = ExtractionEnv(llm_func=None,
                                data_path='your data path of SQuAD2.0.json', 
                                dataset='SQuAD2.0',
                                lang='en',
                                mode='test',
                                data_split=args.data_split)
            plm = "your model path of Llama-3-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(plm, trust_remote_code=True)
        
        print('Begin RL testing!')
        total_score = 0
        relative_position_score = 0

        for i_episode in tqdm(range(env.dataset_len), desc='Sentences to Paragraph Randomly'):
            state, _, _ = env.reset()
            sentences = env.sentence_list
            paragraph = env.paragraph
            rl_indices = []
            for i_step in range(args.max_step):
                cond, choices = state
                action = random.randint(0, len(choices) - 1)
                rl_indices.append(env.mapping[choices[action]])
                next_state, reward, done = env.step(cond, action, choices, rl_indices)
                state = next_state
                if done:
                    break
            
            predict = env.return_cond()
            score = 1 if rl_indices == env.target else 0
            total_score += score
            relative_position = count_consistent_order_optimized(env.target, rl_indices) / ((len(env.target)*(len(env.target)-1))//2)
            relative_position_score += relative_position
        
        print('CAC: ', total_score / env.dataset_len) 
        print('SOC: ', relative_position_score / env.dataset_len)
        
    elif args.test_mode == 'prompt_no_icl' or args.test_mode == 'prompt_icl':
        if args.task_type == 'sentences_to_paragraph_zh':
            env = ExtractionEnv(llm_func=qwen25_14B_api,
                                data_path='your data path of HacRED.json', 
                                dataset='HacRED',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
        elif args.task_type == 'sentences_to_paragraph_en':
            env = ExtractionEnv(llm_func=llama3_8B_api,
                                data_path='your data path of SQuAD2.0.json', 
                                dataset='SQuAD2.0',
                                lang='en',
                                mode='test',
                                data_split=args.data_split)
        
        total_score = 0
        relative_position_score = 0
        
        for i_episode in tqdm(range(env.dataset_len), desc='Sentences to Paragraph Sequentially'):
            state, _, _ = env.reset()
            sentences = env.sentence_list
            paragraph = env.paragraph
            rl_indices = []
            for i_step in range(args.max_step):
                cond, choices = state
                action = env.choose(cond, choices, prompt_mode=args.test_mode)
                if action < 0 or action >= len(choices):
                    action = random.randint(0, len(choices)-1)
                rl_indices.append(env.mapping[choices[action]])
                next_state, reward, done = env.step(cond, action, choices, rl_indices)
                state = next_state
                if done:
                    break
            
            predict = env.return_cond()
            score = 1 if rl_indices == env.target else 0
            total_score += score
            relative_position = count_consistent_order_optimized(env.target, rl_indices) / ((len(env.target)*(len(env.target)-1))//2)
            relative_position_score += relative_position
        
        print('CAC: ', total_score / env.dataset_len) 
        print('SOC: ', relative_position_score / env.dataset_len)