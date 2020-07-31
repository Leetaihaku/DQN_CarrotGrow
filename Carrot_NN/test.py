import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random

num_states = 3
num_actions = 4

model = nn.Sequential()
model.add_module('fc1', nn.Linear(num_states,32))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(32, 32))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(32, num_actions))

print(model)

for _ in range(1):
    init_carrot = np.array([0.0])
    init_humid = np.random.uniform(low=0, high=7, size=1)
    init_temp = np.random.uniform(low=-30.0, high=30.0, size=1)
    init_state = np.array([init_carrot, init_humid, init_temp])

    init_state = torch.from_numpy(init_state)
    init_state = torch.squeeze(init_state)
    init_state = model(init_state.float())

    print(init_state)
    print(torch.argmax(init_state))
