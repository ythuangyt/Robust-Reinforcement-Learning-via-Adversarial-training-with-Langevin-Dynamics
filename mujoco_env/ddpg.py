import torch
from torch.optim import Adam
from torch.optim import RMSprop
from torch.optim import SGLD
from torch.optim import ExtraAdam
from torch.autograd import Variable
import torch.nn.functional as F
from network import Critic, Actor
import os
import numpy as np
from utils import RunningMeanStd
from replay_memory import ReplayMemory, Transition


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def sgld_update(target, source, beta):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + param.data * beta)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def normalize(x, stats, device):
    if stats is None:
        return x
    return (x - torch.Tensor(stats.mean).to(device)) / torch.Tensor(stats.var).sqrt().to(device)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class DDPG:
    def __init__(self, beta, epsilon, learning_rate, gamma, tau, hidden_size_dim0, hidden_size_dim1, num_inputs, action_space, train_mode, alpha, replay_size,
                 optimizer, two_player, normalize_obs=True, normalize_returns=False, critic_l2_reg=1e-2):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = False
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.device = torch.device('cpu')
            self.Tensor = torch.FloatTensor

        self.alpha = alpha
        self.train_mode = train_mode

        self.num_inputs = num_inputs
        self.action_space = action_space
        self.critic_l2_reg = critic_l2_reg

        self.actor = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
        self.adversary = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
        if self.train_mode:
            self.actor_target = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.actor_bar = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.actor_outer = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            if(optimizer == 'SGLD'):
                self.actor_optim = SGLD(self.actor.parameters(), lr=1e-4, noise=epsilon, alpha=0.999)
            elif(optimizer == 'RMSprop'):
                self.actor_optim = RMSprop(self.actor.parameters(), lr=1e-4, alpha=0.999)
            else:
                self.actor_optim = ExtraAdam(self.actor.parameters(), lr=1e-4)

            self.critic = Critic(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.critic_target = Critic(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=1e-3, weight_decay=critic_l2_reg)

            self.adversary_target = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.adversary_bar = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            self.adversary_outer = Actor(hidden_size_dim0, hidden_size_dim1, self.num_inputs, self.action_space).to(self.device)
            if(optimizer == 'SGLD'):
                self.adversary_optim = SGLD(self.adversary.parameters(), lr=1e-4, noise=epsilon, alpha=0.999)
            elif(optimizer == 'RMSprop'):
                self.adversary_optim = RMSprop(self.adversary.parameters(), lr=1e-4, alpha=0.999)
            else:
                self.adversary_optim = ExtraAdam(self.adversary.parameters(), lr=1e-4)
			
            hard_update(self.adversary_target, self.adversary)  # Make sure target is with the same weight
            hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
            hard_update(self.critic_target, self.critic)

        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns
        self.optimizer = optimizer
        self.two_player = two_player
        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=num_inputs)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = ReplayMemory(replay_size)
       
    def eval(self):
        self.actor.eval()
        self.adversary.eval()
        if self.train_mode:
            self.critic.eval()

    def train(self):
        self.actor.train()
        self.adversary.train()
        if self.train_mode:
            self.critic.train()

    def select_action(self, state, action_noise=None, param_noise=None, mdp_type='mdp'):
        state = normalize(Variable(state).to(self.device), self.obs_rms, self.device)

        if mdp_type != 'mdp':
            
            if(self.optimizer == 'SGLD' and self.two_player):
                mu = self.actor_outer(state)
            else:
                mu = self.actor(state)
            mu = mu.data
            if action_noise is not None:
                mu += self.Tensor(action_noise()).to(self.device)

            mu = mu.clamp(-1, 1) * (1 - self.alpha)

            if(self.optimizer == 'SGLD' and self.two_player):
                adv_mu = self.adversary_outer(state)
            else:
                adv_mu = self.adversary(state)
            adv_mu = adv_mu.data.clamp(-1, 1) * self.alpha
            mu += adv_mu
            
        else:
 
            if(self.optimizer == 'SGLD' and self.two_player):
                mu = self.actor_outer(state)
            else:
                mu = self.actor(state)

            mu = mu.data
            if action_noise is not None:
                mu += self.Tensor(action_noise()).to(self.device)

            mu = mu.clamp(-1, 1)

        return mu

    def update_robust_non_flip(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch,
                      mdp_type, robust_update_type):
        # TRAIN CRITIC
        if robust_update_type == 'full':            
            next_action_batch = (1 - self.alpha) * self.actor_target(next_state_batch) \
                                    + self.alpha * self.adversary_target(next_state_batch)

            next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
            expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

            self.critic_optim.zero_grad()
            state_action_batch = self.critic(state_batch, action_batch)

            value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
            value_loss.backward()
            self.critic_optim.step()
            value_loss = value_loss.item()
        else:
            value_loss = 0
        
        # TRAIN ADVERSARY
        self.adversary_optim.zero_grad() 
        with torch.no_grad():
            if(self.optimizer == 'SGLD' and self.two_player):
                real_action = self.actor_outer(next_state_batch)
            else: 
                real_action = self.actor_target(next_state_batch)
        action = (1 - self.alpha) * real_action + self.alpha * self.adversary(next_state_batch)
        adversary_loss = self.critic(state_batch, action)
        adversary_loss = adversary_loss.mean()
        adversary_loss.backward()
        self.adversary_optim.step()
        adversary_loss = adversary_loss.item()
            
        # TRAIN ACTOR
        self.actor_optim.zero_grad()
        with torch.no_grad():
            if(self.optimizer == 'SGLD' and self.two_player):
                adversary_action = self.adversary_outer(next_state_batch)
            else:
                adversary_action = self.adversary_target(next_state_batch)
        action = (1 - self.alpha) * self.actor(next_state_batch) + self.alpha * adversary_action
        policy_loss = -self.critic(state_batch, action)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        policy_loss = policy_loss.item()
           
        return value_loss, policy_loss, adversary_loss
  
    def update_robust_flip(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch, adversary_update,
                      mdp_type, robust_update_type):
        # TRAIN CRITIC
        if robust_update_type == 'full':
           
            next_action_batch = (1 - self.alpha) * self.actor_target(next_state_batch) \
                                    + self.alpha * self.adversary_target(next_state_batch)

            next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
            expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

            self.critic_optim.zero_grad()

            state_action_batch = self.critic(state_batch, action_batch)

            value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
            value_loss.backward()
            self.critic_optim.step()
            value_loss = value_loss.item()
        else:
            value_loss = 0

        if adversary_update:
            # TRAIN ADVERSARY
            self.adversary_optim.zero_grad()
           
            with torch.no_grad():
                real_action = self.actor_target(next_state_batch)
            action = (1 - self.alpha) * real_action + self.alpha * self.adversary(next_state_batch)
            adversary_loss = self.critic(state_batch, action)

            adversary_loss = adversary_loss.mean()
            adversary_loss.backward()
            self.adversary_optim.step()
            adversary_loss = adversary_loss.item()
            policy_loss = 0
        else:
            # TRAIN ACTOR
            self.actor_optim.zero_grad()
            with torch.no_grad():
                adversary_action = self.adversary_target(next_state_batch)
            action = (1 - self.alpha) * self.actor(next_state_batch) + self.alpha * adversary_action
            policy_loss = -self.critic(state_batch, action)

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()
            policy_loss = policy_loss.item()
            adversary_loss = 0

        return value_loss, policy_loss, adversary_loss
  
    def update_non_robust(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        
        # TRAIN CRITIC

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        # TRAIN ACTOR
        self.actor_optim.zero_grad()
        action = self.actor(next_state_batch)
        policy_loss = -self.critic(state_batch, action)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        policy_loss = policy_loss.item()
        adversary_loss = 0

        return value_loss.item(), policy_loss, adversary_loss

    def store_transition(self, state, action, mask, next_state, reward):
        B = state.shape[0]
        for b in range(B):
            self.memory.push(state[b], action[b], mask[b], next_state[b], reward[b])
            if self.normalize_observations:
                self.obs_rms.update(state[b].cpu().numpy())
            if self.normalize_returns:
                self.ret = self.ret * self.gamma + reward[b]
                self.ret_rms.update(np.array([self.ret]))
                if mask[b] == 0:  # if terminal is True
                    self.ret = 0

    def update_parameters(self, batch_size, sgld_outer_update, mdp_type='mdp', exploration_method='mdp'):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        if mdp_type != 'mdp':
            robust_update_type = 'full'
        elif exploration_method != 'mdp':
            robust_update_type = 'adversary'
        else:
            robust_update_type = None

        state_batch = normalize(Variable(torch.stack(batch.state)).to(self.device), self.obs_rms, self.device)
        action_batch = Variable(torch.stack(batch.action)).to(self.device)
        reward_batch = normalize(Variable(torch.stack(batch.reward)).to(self.device).unsqueeze(1), self.ret_rms, self.device)
        mask_batch = Variable(torch.stack(batch.mask)).to(self.device).unsqueeze(1)
        next_state_batch = normalize(Variable(torch.stack(batch.next_state)).to(self.device), self.obs_rms, self.device)

        if self.normalize_returns:
            reward_batch = torch.clamp(reward_batch, -self.cliprew, self.cliprew)

        value_loss = 0
        policy_loss = 0
        adversary_loss = 0
        
        if robust_update_type is not None:
           
            _value_loss, _policy_loss, _adversary_loss = self.update_robust_non_flip(state_batch, action_batch, reward_batch,
                                                                            mask_batch, next_state_batch,                                                                                                                            mdp_type, robust_update_type)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss
           
        if robust_update_type != 'full':
            _value_loss, _policy_loss, _adversary_loss = self.update_non_robust(state_batch, action_batch,
                                                                                reward_batch,
                                                                                mask_batch, next_state_batch)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss
        
        if(self.optimizer == 'SGLD' and self.two_player):   
            self.sgld_inner_update()
        self.soft_update()
        if(sgld_outer_update and self.optimizer == 'SGLD' and self.two_player):
            self.sgld_outer_update()

        return value_loss, policy_loss, adversary_loss

    def initialize(self):
        hard_update(self.actor_bar, self.actor_outer)
        hard_update(self.adversary_bar, self.adversary_outer)
        hard_update(self.actor, self.actor_outer)
        hard_update(self.adversary, self.adversary_outer)

    def sgld_inner_update(self): #target source
        sgld_update(self.actor_bar, self.actor, self.beta)
        sgld_update(self.adversary_bar, self.adversary, self.beta)

    def sgld_outer_update(self): #target source
        sgld_update(self.actor_outer, self.actor_bar, self.beta)
        sgld_update(self.adversary_outer, self.adversary_bar, self.beta)

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.adversary_target, self.adversary, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

