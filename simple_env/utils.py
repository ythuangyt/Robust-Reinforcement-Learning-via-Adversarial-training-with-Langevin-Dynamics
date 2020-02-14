import os
import torch
import numpy as np

def load_model(agent, basedir=None):
    agent_path = "{}/agent_".format(basedir) + str(4900)
    adversary_path = "{}/adversary_".format(basedir) + str(4900)

    print('Loading models from {} {}'.format(agent_path, adversary_path))
    agent.agent.load_state_dict(torch.load(agent_path, map_location=lambda storage, loc: storage))
    agent.adversary.load_state_dict(torch.load(adversary_path, map_location=lambda storage, loc: storage))
