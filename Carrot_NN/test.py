import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
'''
num_states = 3
num_actions = 4
learning_rate = 0.01'''

'''

model = nn.Sequential()
model.add_module('fc1', nn.Linear(num_states, 32))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(32, 32))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(32, num_actions))

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)






init_carrot = np.array([0.0])
init_humid = np.random.uniform(low=0, high=7, size=1)
init_temp = np.random.uniform(low=-30.0, high=30.0, size=1)
init_state = np.array([init_carrot, init_humid, init_temp])
init_state = torch.from_numpy(init_state).double()
init_state = torch.squeeze(init_state)

init_carrot = np.array([0.0])
init_humid = np.random.uniform(low=0, high=7, size=1)
init_temp = np.random.uniform(low=-30.0, high=30.0, size=1)
init_state1 = np.array([init_carrot, init_humid, init_temp])
init_state1 = torch.from_numpy(init_state1).double()
init_state1 = torch.squeeze(init_state1)

init_state = init_state.float()
init_state1 = init_state1.float()

print(init_state.type())
print(init_state1.type())

init = []

init.append(None)
init[0] = np.array([init_state[0],init_state[1],init_state[2]])
init.append(None)
init[1] = np.array([init_state1[0],init_state1[1],init_state1[2]])


#[TEST(state1=tensor(0.), state2=tensor(3.4062), state3=tensor(-21.3212)), TEST(state1=tensor(0.), state2=tensor(4.3320), state3=tensor(12.7644))]
print(init)

for _ in range(2):
    init[_] = torch.from_numpy(init[_])

    model.eval()
    print('추론모드 전환')

    state_result = model(init[_])
    print(state_result)
    print(state_result[2])
    print(max(state_result))




#print('모델 raw 반환 값\t', model(init))



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


a = 1.05
b = 1.22
c = []
c.append(a)
c.append(b)

print(c)
print(torch.tensor([a]))
'''
'''
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state = env.reset()
    print(state)
    
'''
'''if self.Carrot < -2.0:
            
            reward = -2.0
        elif self.Carrot >= -2.0 and self.Carrot < -1.6:
            
            reward = -1.6
        elif self.Carrot >= -1.6 and self.Carrot < -1.2:
            
            reward = -1.2
        elif self.Carrot >= -1.2 and self.Carrot < -0.8:
            
            reward = -0.8
        elif self.Carrot >= -0.8 and self.Carrot < -0.4:
            
            reward = -0.4
        elif self.Carrot >= -0.4 and self.Carrot < 0.4:
            
            reward = 0.0
        elif self.Carrot >= 0.4 and self.Carrot < 0.8:
            
            reward = 0.4
        elif self.Carrot >= 0.8 and self.Carrot < 1.2:
            
            reward = 1.4
        elif self.Carrot >= 1.2 and self.Carrot < 1.6:
            
            reward = 1.6
        elif self.Carrot >= 1.6 and self.Carrot < 2.0:
            
            reward = 1.8
        elif self.Carrot == 2.0:
            reward = 2.0'''

'''
def HP_calculation(humid, temp):
    # 당근 체력 = 수분량 적정도 50% + 온도 적정도 50%
    carrot = (0.5 - 0.5 * abs(7 - humid) / 7) + Temp_calculation()
    return carrot


def Temp_calculation(x):
    return

if __name__ == '__main__':
    print(Temp_calculation(18))'''

'''a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
a = torch.from_numpy(a).float()
b = torch.from_numpy(b).float()
ab = a+b
ab = torch.tensor().stack()
print(a.max(1)[0])'''

done = torch.tensor([[True],
                     [False]])
print(done)

print(~done)