import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
import copy

'''NODES = 24
PATH = '.\saved_model\Beta-17.pth'

def modeling_NN():
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(2, NODES))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(NODES, NODES))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(NODES, 4))
    return model

Q = modeling_NN()
Q.load_state_dict(torch.load(PATH))

Q.eval()

print(Q(torch.squeeze(torch.tensor([1,1]))))'''
reward = 0.1
reward2 = 0.2
reward2 = torch.tensor(reward2)
reward = np.array([reward])
reward = torch.from_numpy(reward)
reward = reward.float()
print(reward.shape)
print(reward2.shape)