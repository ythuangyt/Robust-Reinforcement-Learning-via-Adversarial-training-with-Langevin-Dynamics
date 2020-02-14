import copy
import torch
import torch.nn as nn
import numpy as np

from torch.optim.optimizer import Optimizer, required

class OptimisticAdam(Optimizer):
    '''
    >>> implementation of TRAINING GANS WITH OPTIMISM
    '''

    def __init__(self, params, lr = required, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 0.):

        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: %1.1e'%lr)
        if weight_decay < 0.0:
            raise ValueError('Invalid weight decay value: %1.1e'%weight_decay)

        default = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)

        super(OptimisticAdam, self).__init__(params, default)

    def __setstate__(self, state):

        super(OptimisticAdam, self).__setstate__(state)

    def step(self, closure = None):

        loss = None
        if closure != None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0.:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]

                if 'step' not in param_state:
                    param_state['step'] = 0
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                if 'exp_avg_sq' not in param_state:
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)

                if param_state['step'] != 0:
                    bias_corr1 = 1. - beta1 ** param_state['step']
                    bias_corr2 = 1. - beta2 ** param_state['step']
                    step_size = lr * np.sqrt(bias_corr2) / bias_corr1
                    p.data.addcdiv_(step_size, param_state['exp_avg'], torch.sqrt(param_state['exp_avg_sq'] + eps))

                param_state['step'] += 1
                times = 1 if param_state['step'] == 1 else 2

                param_state['exp_avg'].mul_(beta1).add_(1. - beta1, d_p)
                param_state['exp_avg_sq'].mul_(beta2).addcmul_(1. - beta2, d_p, d_p)

                bias_corr1 = 1. - beta1 ** param_state['step']
                bias_corr2 = 1. - beta2 ** param_state['step']
                step_size = lr * np.sqrt(bias_corr2) / bias_corr1
                p.data.addcdiv_(-step_size * times, param_state['exp_avg'], torch.sqrt(param_state['exp_avg_sq'] + eps))

        return loss

class ExtraAdam(Optimizer):
    '''
    >>> implementation of A VARIATIONAL INEQUALITY PERSPECTIVE ON GENERATIVE ADVERSARIAL NETWORKS
    '''

    def __init__(self, params, lr = required, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 0.):

        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: %1.1e'%lr)
        if weight_decay < 0.0:
            raise ValueError('Invalid weight decay value: %1.1e'%weight_decay)

        default = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)

        super(ExtraAdam, self).__init__(params, default)

    def __setstate__(self, state):

        super(ExtraAdam, self).__setstate__(state)

    def step(self, closure = None):

        loss = None
        if closure != None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0.:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]

                if 'step' not in param_state:
                    param_state['step'] = 0
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                if 'exp_avg_sq' not in param_state:
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                if 'memory' not in param_state:
                    param_state['memory'] = torch.zeros_like(p.data)

                param_state['step'] += 1
                if param_state['step'] % 2 == 1:
                    param_state['memory'] = copy.copy(p.data)

                param_state['exp_avg'].mul_(beta1).add_(1. - beta1, d_p)
                param_state['exp_avg_sq'].mul_(beta2).addcmul_(1. - beta2, d_p, d_p)

                bias_corr1 = 1. - beta1 ** param_state['step']
                bias_corr2 = 1. - beta2 ** param_state['step']
                step_size = lr * np.sqrt(bias_corr2) / bias_corr1
                if param_state['step'] % 2 == 0:
                    p.data = param_state['memory']
                p.data.addcdiv_(-step_size, param_state['exp_avg'], torch.sqrt(param_state['exp_avg_sq'] + eps))
