import argparse, math, os
import numpy as np
import gym
import gym_simple
from gym import wrappers
import random
import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from normalized_actions import NormalizedActions
from torch.distributions import uniform
import random
import matplotlib.pyplot as plt
import pickle
import ast
from reinforce_continuous import REINFORCE

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='simple-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=500, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=5000, metavar='N',
                    help='number of episodes (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=16, metavar='N',
                    help='number of episodes (default: 16)')
parser.add_argument('--delta', type=float, default=0, metavar='N',
                    help='delta (default: 0)')
parser.add_argument('--optimizer', default='SGLD', choices=['SGLD', 'RMSprop', 'E_SGLD', 'ExtraAdam'] )
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=100, help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
parser.add_argument('--lr', type=float)
parser.add_argument('--thermal_noise', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--two_player', type=ast.literal_eval)
parser.add_argument('--Kt', type=int)
parser.add_argument('--beta', type=float)

args = parser.parse_args()

env_name = args.env_name
env = gym.envs.make(env_name)
eval_env = gym.envs.make(env_name)

if args.display:
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space, optimizer=args.optimizer, 
                lr=args.lr, thermal_noise=args.thermal_noise, beta=args.beta, alpha=args.alpha, two_player=args.two_player)

if args.two_player:
    base_dir = os.getcwd() + '/models_TwoPlayer/' + args.env_name + '/' 
else:
    base_dir = os.getcwd() + '/models_OnePlayer/' + args.env_name + '/'

if args.optimizer == 'SGLD':
    base_dir += args.optimizer + '_thermal_' + str(args.thermal_noise) + '/'
elif(args.optimizer == 'E_SGLD'):
    base_dir += args.optimizer + '_thermal_' + str(args.thermal_noise) + '/'
else:
    base_dir += args.optimizer + '/'
base_dir += 'delta_' + str(args.delta) + '/lr_' + str(args.lr) + '/'
base_dir += 'beta_' + str(args.beta) + '/Kt_' + str(args.Kt) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number)

os.makedirs(base_dir)

noise = uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))
results_dict = {'eval_rewards': [], 'train_rewards': []}

eval_reward = 0

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    eval_state = torch.Tensor([eval_env.reset()])

    agent_log_probs = []
    adv_log_probs = []
    episode_rewards = []
    
    # warmup steps for SGLD
    if (args.optimizer == 'SGLD'):
        kt = args.Kt
        agent.initialize()
    # normal setup for RMSPROP and SGLD
    elif(args.optimizer == 'E_SGLD'):
        kt = 1
        agent.initialize()        
    else:
        kt = 1

    for k in range(kt):
        sgld_outer_update = (k == kt - 1)
        for t in range(args.num_steps):
            action, agent_log_prob, adv_log_prob = agent.select_action(state)
            if random.random() < args.delta:
                action = noise.sample(action.shape).view(action.shape)
            action = action.cpu()
            next_state, reward, done, _ = env.step(action.numpy()[0])
            agent_log_probs.append(agent_log_prob)
            adv_log_probs.append(adv_log_prob)
            episode_rewards.append(reward)
            state = torch.Tensor([next_state])
            if done:
               break
        
        agent.update_parameters(episode_rewards, agent_log_probs, adv_log_probs, args.gamma, sgld_outer_update=sgld_outer_update)

    print("Episode: {}, reward: {}".format(i_episode, np.sum(episode_rewards)/kt))
    results_dict['train_rewards'].append((i_episode, np.sum(episode_rewards)/kt))
            
    # evaluation stage, with different environment from training stage
    with torch.no_grad():
        for t in range(args.num_steps):
            action, _, _ = agent.select_action(eval_state)
            if random.random() < args.delta:
                action = noise.sample(action.shape).view(action.shape)
            action = action.cpu()
            next_eval_state, reward, done, _ = eval_env.step(action.numpy()[0])
            eval_reward += reward

            next_eval_state = torch.Tensor([next_eval_state])

            eval_state = next_eval_state
            if done:
                break
        results_dict['eval_rewards'].append((i_episode, eval_reward))
        eval_state = torch.Tensor([eval_env.reset()])
        eval_reward = 0
    if i_episode%args.ckpt_freq == 0:
        torch.save(agent.agent.state_dict(), os.path.join(base_dir, 'agent_' + str(i_episode)))
        torch.save(agent.adversary.state_dict(), os.path.join(base_dir, 'adversary_'+str(i_episode)))
    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)

env.close()
