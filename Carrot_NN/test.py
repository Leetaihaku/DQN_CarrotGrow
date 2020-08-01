import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random

'''num_states = 3
num_actions = 4
learning_rate = 0.01

init_carrot = np.array([0.0])
init_humid = np.random.uniform(low=0, high=7, size=1)
init_temp = np.random.uniform(low=-30.0, high=30.0, size=1)
init_state = np.array([init_carrot, init_humid, init_temp])
init_state = torch.from_numpy(init_state).double()
init_state = torch.squeeze(init_state)
print(init_state)

model = nn.Sequential()
model.add_module('fc1', nn.Linear(num_states, 32))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(32, 32))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(32, num_actions))

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

print('모델 raw 반환 값\t', model(init_state.float()))
print('모델결과 아그맥스\t', torch.argmax(model(init_state.float())))
print('모델 뷰(1x1)\t\t', torch.argmax(model(init_state.float())).view(1, 1))
print('탐험 행동\t\t', torch.LongTensor([[random.randrange(num_actions)]]))

print('이용 타입\t\t', torch.argmax(model(init_state.float())).view(1, 1).type())
print('탐험 타입\t\t', torch.LongTensor([[random.randrange(num_actions)]]).type())

exploit = torch.argmax(model(init_state.float())).view(1, 1)
explore = torch.LongTensor([[random.randrange(num_actions)]])

if exploit < 4:
    print('exploit success')
else:
    print('exploit fail')

if exploit < 4:
    print('explore success')
else:
    print('explore fail')
'''

v = torch.tensor([[1,2,3],
                  [7,8,9],
                  [10,11,12]])
vv = torch.gather(v,0,torch.tensor([[0,1],
                                    [1,2],
                                    [2,2]]))
print(v)
print(vv)

a = torch.tensor([[1]])
print(a.item())