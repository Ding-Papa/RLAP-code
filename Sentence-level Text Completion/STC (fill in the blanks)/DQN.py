import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from Environment import ExtractionEnv
import argparse
import numpy as np
from tqdm import tqdm
from RL_utils.dqn import DQN
import math
from llm_server import qwen25_14B_api

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=888, help="seed")
parser.add_argument("--plm", type=str, default='your model path of GTE', help='pretrain language model')
parser.add_argument("--exploration_update", type=int, default=100, help='how many step to update exploration ratio')
parser.add_argument("--max_step", type=int, default=15, help='max step on a single episode')
parser.add_argument("--num_episode", type=int, default=10, help='sample episode number')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate for the DQN model')
parser.add_argument("--do_train", type=bool, default=False, help='whether train')
parser.add_argument("--do_test", type=bool, default=False, help='whether test')
parser.add_argument("--test_mode", type=str, choices=['rl', 'random', 'sequence', 'prompt_no_icl', 'prompt_icl'], default='rl')
parser.add_argument("--weight_file", type=str, help='path to save rl model weight')
parser.add_argument('--task_type', type=str, choices=['sentences_to_paragraph_zh', 'sentences_to_paragraph_en', 'complete_sentence_blanks'])
parser.add_argument('--buf_size', type=int, default=5000, help='the size of the memory size')
parser.add_argument('--data_split', type=float, default=0.8)
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
        "batch_size": 16,
        "seed": args.seed,
    }
  
    tokenizer = AutoTokenizer.from_pretrained(plm)
    data_path = './Sentence-level Text Completion/STC (fill in the blanks) /dataset/cmrc2019.json'
    if args.task_type == 'complete_sentence_blanks':
        env = ExtractionEnv(llm_func=qwen25_14B_api,
                            data_path=data_path, 
                            dataset='cmrc2019',
                            lang='zh',
                            mode='train',
                            data_split=args.data_split)

    tot_step = args.num_episode * env.dataset_len
    n = math.log(0.05 / 0.9) / math.log(0.95)
    explore_update = int(tot_step // n)
    print('Explore update:', explore_update)
    agent = DQN(plm=plm,
                epsilon=0.9, 
                tokenizer=tokenizer, 
                gamma=0.5,
                buf_sz=args.buf_size,
                batch_sz=16, 
                lr=lr, 
                explore_update=explore_update)
    


    print('Begin RL training!')

    rewards = []
    moving_average_rewards = []
    ep_steps = []
    for i_episode in tqdm(range(tot_step), desc='Training RL agent'):
        state, _, _ = env.reset()
        ep_reward = 0
        for i_step in range(args.max_step):
            cond, choices, blanks = state
            action = agent.select_action(cond, choices)
            action = torch.argmax(action)
            next_state, reward, done = env.step(cond, action, choices, blanks)
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
    data_path = './Sentence-level Text Completion/STC (fill in the blanks) /dataset/cmrc2019.json'
    if args.test_mode == 'rl':   
        if args.task_type == 'complete_sentence_blanks':
            env = ExtractionEnv(llm_func=qwen25_14B_api,
                                data_path=data_path, 
                                dataset='cmrc2019',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
        
        tokenizer = AutoTokenizer.from_pretrained(plm)
        
        agent = DQN(plm=plm,
                    epsilon=0, 
                    tokenizer=tokenizer, 
                    gamma=0.5,
                    buf_sz=10000,
                    batch_sz=16, 
                    lr=lr, 
                    explore_update = 1e10,
                    mode='test')
        agent.load_weight(f'weight/{args.weight_file}')

        print('Begin RL testing!')
        question_count = 0
        question_right_count = 0
        total_right_count = 0
        total_count = 0

        for i_episode in tqdm(range(env.dataset_len), desc='Complete Sentence Blanks through RL agent'):
            state, _, _ = env.reset()
            total_count += 1
            total_reward = 0
            for i_step in range(args.max_step):
                question_count += 1
                cond, choices, blanks = state
                action = agent.select_action(cond, choices)
                action = torch.argmax(action)
                next_state, reward, done = env.step(cond, action, choices, blanks)
                if reward == 1:
                    question_right_count += 1
                    total_reward += 1
                state = next_state
                if done:
                    break
                
            total_right_count += total_reward == len(env.answers)
        
        if question_count == 0:
            print('question_count: 0, cannot calculate BAC')
        else:
            print('BAC: ', question_right_count / question_count)
        
        if total_count == 0:
            print('total_count: 0, cannot calculate CAC')
        else:
            print('CAC: ', total_right_count / total_count)
    
    elif args.test_mode == 'random':
        if args.task_type == 'complete_sentence_blanks':
            env = ExtractionEnv(llm_func=qwen25_14B_api,
                                data_path=data_path, 
                                dataset='cmrc2019',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
        
        print('Begin RL testing!')
        question_count = 0
        question_right_count = 0
        total_right_count = 0
        total_count = 0
        
        for i_episode in tqdm(range(env.dataset_len), desc='Complete Sentence Blanks Randomly'):
            state, _, _ = env.reset()
            total_reward = 0
            total_count += 1
            for i_step in range(args.max_step):
                question_count += 1
                cond, choices, blanks = state
                action = random.randint(0, len(choices) - 1)
                next_state, reward, done = env.step(cond, action, choices, blanks)
                if reward == 1:
                    question_right_count += 1
                    total_reward += 1
                state = next_state
                if done:
                    break
            
            total_right_count += total_reward == len(env.answers)
        
        if question_count == 0:
            print('question_count: 0, cannot calculate BAC')
        else:
            print('BAC: ', question_right_count / question_count)
        
        if total_count == 0:
            print('total_count: 0, cannot calculate CAC')
        else:
            print('CAC: ', total_right_count / total_count)
        
    elif args.test_mode == 'sequence':
        if args.task_type == 'complete_sentence_blanks':
            env = ExtractionEnv(llm_func=qwen25_14B_api,
                                data_path=data_path, 
                                dataset='cmrc2019',
                                lang='zh',
                                mode='test',
                                data_split=args.data_split)
        
        print('Begin RL testing!')
        question_count = 0
        question_right_count = 0
        total_right_count = 0
        total_count = 0
        
        for i_episode in tqdm(range(env.dataset_len), desc='Complete Sentence Blanks Sequentially'):
            state, _, _ = env.reset()
            total_reward = 0
            total_count += 1
            for i_step in range(args.max_step):
                question_count += 1
                cond, choices, blanks = state
                action = 0
                next_state, reward, done = env.step(cond, action, choices, blanks)
                if reward == 1:
                    question_right_count += 1
                    total_reward += 1
                state = next_state
                if done:
                    break
            
            total_right_count += total_reward == len(env.answers)
        
        if question_count == 0:
            print('question_count: 0, cannot calculate BAC')
        else:
            print('BAC: ', question_right_count / question_count)
        
        if total_count == 0:
            print('total_count: 0, cannot calculate CAC')
        else:
            print('CAC: ', total_right_count / total_count)