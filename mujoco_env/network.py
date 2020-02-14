import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, hidden_size_dim0, hidden_size_dim1, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size_dim0)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.linear2 = nn.Linear(hidden_size_dim0, hidden_size_dim1)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)

        self.mu = nn.Linear(hidden_size_dim1, num_outputs)
        torch.nn.init.uniform_(self.mu.weight, a=-3e-3, b=3e-3)

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size_dim0, hidden_size_dim1, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs + num_outputs, hidden_size_dim0)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.ln1 = nn.LayerNorm(hidden_size_dim0)

        self.linear2 = nn.Linear(hidden_size_dim0, hidden_size_dim1)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)
        self.ln2 = nn.LayerNorm(hidden_size_dim1)

        self.V = nn.Linear(hidden_size_dim1, 1)
        torch.nn.init.uniform_(self.V.weight, a=-3e-3, b=3e-3)

    def forward(self, inputs, actions):
        x = torch.cat((inputs, actions), 1)
        x = self.linear1(x)
        x = self.ln1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = torch.tanh(x)
        V = self.V(x)
        return V
