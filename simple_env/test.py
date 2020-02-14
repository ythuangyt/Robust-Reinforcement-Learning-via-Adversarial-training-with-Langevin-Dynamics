import argparse
import os
import gym
import gym_simple
import numpy as np
import pickle
import copy
import random
import ast
import torch
from torch.distributions import uniform

from reinforce_continuous import REINFORCE
from utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--eval_type', default='model',
                    choices=['model', 'model_noise'])
parser.add_argument('--hidden_size', type=int, default=16, metavar='N',
                    help='number of neurons in the hidden layers (default: 64)')
parser.add_argument('--num_steps', type=int, default=500)
parser.add_argument('--two_player', type=ast.literal_eval)
args = parser.parse_args()

if(args.two_player):
    base_dir = os.getcwd() + '/models_TwoPlayer/'
    alpha = 0.1
else:
    base_dir = os.getcwd() + '/models_OnePlayer/'
    alpha = 0


def eval_model(_env, delta):
    total_reward = 0
    with torch.no_grad():
        state = torch.Tensor([_env.reset()])
        for t in range(args.num_steps):
            action, _, _ = agent.select_action(state)
            if random.random() < delta:
                action = noise.sample(action.shape).view(action.shape)
            action = action.cpu()
            state, reward, done, _ = _env.step(action.numpy()[0])
            total_reward += reward

            state = torch.Tensor([state])
            if done:
                break
    return total_reward


test_episodes = 20
for env_name in ['simple-v0']:#os.listdir(base_dir):
    env = gym.envs.make(env_name)
    
    agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space, optimizer=0, 
                lr=1e-4, thermal_noise=0, beta=0.9, alpha=alpha, two_player=args.two_player)

    noise = uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))

    #basic_bm = copy.deepcopy(env.env.env.model.body_mass.copy())

    env_dir = base_dir + env_name + '/SGLD_thermal_0.0001/delta_0.2/lr_0.0001/'
#   for test_dir in ['ExtraAdam/delta_0.0/lr_0.001/', 'RMSprop/delta_0.0/lr_0.001/', 'SGLD_thermal_0.0001/delta_0.0/lr_0.0001/',\
#	'ExtraAdam/delta_0.1/lr_0.001/', 'RMSprop/delta_0.1/lr_0.001/', 'SGLD_thermal_0.001/delta_0.1/lr_0.0001/',\
#	'ExtraAdam/delta_0.2/lr_0.0001/', 'RMSprop/delta_0.2/lr_0.0001/', 'SGLD_thermal_1e-05/delta_0.2/lr_0.001/']:
    for test_dir in ['beta_0.5/Kt_1/', 'beta_0.5/Kt_2/', 'beta_0.5/Kt_5/', 'beta_0.9/Kt_1/', 'beta_0.9/Kt_2/', \
	'beta_0.9/Kt_5/', 'beta_1.0/Kt_1/', 'beta_1.0/Kt_2/', 'beta_1.0/Kt_5/']: 
        noise_dir = env_dir + test_dir
        if os.path.exists(noise_dir):
            for subdir in sorted(os.listdir(noise_dir)):
                results = {}
                
                run_number = 0
                dir = noise_dir + subdir #+ '/' + str(run_number)
                print(dir)
                if os.path.exists(noise_dir + subdir):\
                #and not os.path.isfile(noise_dir + subdir + '/results_' + args.eval_type):
                    while os.path.exists(dir):
                        load_model(agent=agent, basedir=dir)
                        agent.eval()

                        for delta in np.linspace(0, 0.4, 20):
                            if delta not in results:
                                results[delta] = []
                            for _ in range(test_episodes):
                                r = eval_model(env, delta)
                                results[delta].append(r)

                        run_number += 1
                        dir = noise_dir + subdir + '/' + str(run_number)
                    with open(noise_dir + subdir + '/results_model', 'wb') as f:
                        pickle.dump(results, f)

env.close()
