import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque



NUM_STATES = 3
NUM_ACTIONS = 4
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
BATCH_SIZE = 32
TRAIN_START = 1000



class Carrot_House  :  # 하우스 환경

    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Carrot = 0.0
        self.Humid = 0
        self.Temp = 0.0
        self.Cumulative_Step = 0
        self.agent = Agent(NUM_STATES, NUM_ACTIONS)

    def step(self, action):
        '''행동진행 => 환경결과'''
        # 물주기
        if action == 0:
            self.supply_water()
        # 온도 올리기
        elif action == 1:
            self.Temp_up()
        # 온도 내리기
        elif action == 2:
            self.Temp_down()
        # 현상유지
        elif action == 3:
            self.Wait()

        self.Carrot = self.HP_calculation(self.Humid, self.Temp)

        # 하루치 수분량 감쇄
        self.Humid -= 1

        # 종료여부
        if self.Cumulative_Step == MAX_STEP:
            done = True
        else:
            done = False

        # 보상
        reward = -0.001

        next_state = np.array([self.Carrot, self.Humid, self.Temp])
        next_state = torch.from_numpy(next_state)
        next_state = next_state.float()
        reward = np.array([reward])
        reward = torch.from_numpy(reward)
        reward = reward.float()
        return torch.tensor(next_state), torch.tensor(reward), done

    def supply_water(self):
        self.Humid = 7

    def Temp_up(self):
        self.Temp += 1.0

    def Temp_down(self):
        self.Temp -= 1.0

    def Wait(self):
        return

    def HP_calculation(self, humid, temp):
        # 온도에 대해서만 차이계산
        if humid > 0:
            carrot = 1.0 - 0.5 * abs(18.0 - temp) / 60
        # 전체 차이계산
        else:
            carrot = 0.5 - 0.5 * abs(18.0 - temp) / 60
        return carrot

    def reset(self):
        '''환경 초기화'''
        init_carrot = np.array([0.0])
        init_humid = np.random.uniform(low=0, high=7, size=1)
        init_temp = np.random.uniform(low=-30.0, high=30.0, size=1)
        init_state = np.array([init_carrot, init_humid, init_temp])
        init_state = torch.from_numpy(init_state)
        init_state = torch.squeeze(init_state, 1)
        return init_state.float()


class DQNAgent:

    def __init__(self, state_size, action_size):
        #상태사이즈, 행동사이즈 정의 및 메모리 생성
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        #Q-Net과 Target-Q-Net 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Target-Q-Net 동기화 >> Q-Net
        self.update_target_model()

        '''(Option)훈련한 Q-Net 로드 하기'''

    def build_model(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(num_states, 32))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(32, 32))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc3', nn.Linear(32, num_actions))
        model.optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
        return model









