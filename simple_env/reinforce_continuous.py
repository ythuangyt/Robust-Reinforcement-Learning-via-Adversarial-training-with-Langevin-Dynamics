import sys
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi]))

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

def sgld_update(target, source, beta):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + param.data * beta)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space, optimizer, lr, thermal_noise, beta, alpha, two_player):
        self.action_space = action_space
        self.agent = Policy(hidden_size, num_inputs, action_space)
        self.agent_bar = Policy(hidden_size, num_inputs, action_space)
        self.agent_outer = Policy(hidden_size, num_inputs, action_space)
        self.adversary = Policy(hidden_size, num_inputs, action_space)
        self.adversary_bar = Policy(hidden_size, num_inputs, action_space)
        self.adversary_outer = Policy(hidden_size, num_inputs, action_space)

        self.optimizer = optimizer
        if(self.optimizer == "RMSprop"):
            self.agent_optimizer = optim.RMSprop(self.agent.parameters(), lr=lr, alpha=0.99)
            self.adversary_optimizer = optim.RMSprop(self.adversary.parameters(), lr=lr, alpha=0.99)
        elif(self.optimizer == "ExtraAdam"):
            self.agent_optimizer = optim.ExtraAdam(self.agent.parameters(), lr=lr)
            self.adversary_optimizer = optim.ExtraAdam(self.adversary.parameters(), lr=lr)
        else:
            self.agent_optimizer = optim.SGLD(self.agent.parameters(), lr=lr, noise=thermal_noise, alpha=0.99)
            self.adversary_optimizer = optim.SGLD(self.adversary.parameters(), lr=lr, noise=thermal_noise, alpha=0.99)
        self.train()

        self.beta = beta
        self.alpha = alpha
        self.two_player = two_player

    def train(self):
        self.agent.train()
        self.adversary.train()
    
    def eval(self):
        self.agent.eval()
        self.adversary.eval()
    
    def select_action(self, state):

        # agent action
        agent_mu, agent_sigma_sq = self.agent(Variable(state))
        agent_sigma_sq = F.softplus(agent_sigma_sq)

        agent_eps = torch.randn(agent_mu.size())
        # calculate the probability
        agent_action = (agent_mu + agent_sigma_sq.sqrt()*Variable(agent_eps)).data
        agent_prob = normal(agent_action, agent_mu, agent_sigma_sq)

        agent_log_prob = agent_prob.log()
        if(self.two_player):
            # adversary action
            adv_mu, adv_sigma_sq = self.adversary(Variable(state))
            adv_sigma_sq = F.softplus(adv_sigma_sq)
            
            adv_eps = torch.randn(adv_mu.size())
            # calculate the probability
            adv_action = (adv_mu + adv_sigma_sq.sqrt()*Variable(adv_eps)).data
            adv_prob = normal(adv_action, adv_mu, adv_sigma_sq)
            
            adv_log_prob = adv_prob.log()
            
            action = (1 - self.alpha) * agent_action + self.alpha * adv_action
        else:
            action = agent_action
            adv_log_prob = 0

        return action, agent_log_prob, adv_log_prob

    def update_parameters(self, rewards, agent_log_probs, adv_log_probs, gamma, sgld_outer_update):
        agent_R = torch.zeros(1, 1)
        agent_loss = 0
        for i in reversed(range(len(rewards))):
            agent_R = gamma * agent_R + rewards[i]
            agent_loss = agent_loss - (agent_log_probs[i]*(Variable(agent_R).expand_as(agent_log_probs[i]))).sum()
        agent_loss = agent_loss / len(rewards)

        self.agent_optimizer.zero_grad()
        agent_loss.backward(retain_graph=True)
        utils.clip_grad_norm_(self.agent.parameters(), 40)
        self.agent_optimizer.step()
        if(self.two_player):
            adv_R = torch.zeros(1, 1)
            adv_loss = 0
            for i in reversed(range(len(rewards))):
                adv_R = gamma * adv_R + rewards[i]
                adv_loss = adv_loss - (adv_log_probs[i]*(Variable(adv_R).expand_as(adv_log_probs[i]))).sum()
            adv_loss = -adv_loss / len(rewards)
            
            self.adversary_optimizer.zero_grad()
            adv_loss.backward(retain_graph=True)
            utils.clip_grad_norm_(self.adversary.parameters(), 40)
            self.adversary_optimizer.step()
        
        if((self.optimizer != 'RMSprop') or (self.optimizer != 'ExtraAdam')):   
            self.sgld_inner_update()
        if(sgld_outer_update and (self.optimizer != 'RMSprop' or self.optimizer != 'ExtraAdam')):
            self.sgld_outer_update()

    def initialize(self):
        hard_update(self.agent_bar, self.agent_outer)
        hard_update(self.agent, self.agent_outer)
        hard_update(self.adversary_bar, self.adversary_outer)
        hard_update(self.adversary, self.adversary_outer)

    def sgld_inner_update(self): #target source
        sgld_update(self.agent_bar, self.agent, self.beta)
        sgld_update(self.adversary_bar, self.adversary, self.beta)

    def sgld_outer_update(self): #target source
        sgld_update(self.agent_outer, self.agent_bar, self.beta)
        sgld_update(self.adversary_outer, self.adversary_bar, self.beta)

