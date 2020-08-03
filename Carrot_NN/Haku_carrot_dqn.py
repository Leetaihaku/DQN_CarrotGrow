import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple

NUM_STATES = 3
NUM_ACTIONS = 4
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
BATCH_SIZE = 32
TRAIN_START = 1000
CAPACITY = 10000
EPISODES = 2
MAX_STEPS = 300
DATA = namedtuple('DATA',('state','action','reward','next_state','done'))

class Carrot_House  :  #  하우스 환경

    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Carrot = 0.0
        self.Humid = 0
        self.Temp = 0.0
        self.Cumulative_Step = 0

    def step(self, action):
        '''행동진행 => 환경결과'''
        #  물주기
        if action == 0:
            self.supply_water()
        #  온도 올리기
        elif action == 1:
            self.Temp_up()
        #  온도 내리기
        elif action == 2:
            self.Temp_down()
        #  현상유지
        elif action == 3:
            self.Wait()

        self.Carrot = self.HP_calculation(self.Humid, self.Temp)

        #  하루치 수분량 감쇄
        self.Humid -= 1

        #  종료여부
        if self.Cumulative_Step == MAX_STEPS:
            done = True
        else:
            done = False

        #  보상
        reward = -0.001

        next_state = np.array([self.Carrot, self.Humid, self.Temp])
        next_state = torch.from_numpy(next_state)
        next_state = next_state.float()
        reward = np.array([reward])
        reward = torch.from_numpy(reward)
        reward = reward.float()
        return next_state, reward, done

    def supply_water(self):
        self.Humid = 7

    def Temp_up(self):
        self.Temp += 1.0

    def Temp_down(self):
        self.Temp -= 1.0

    def Wait(self):
        return

    def HP_calculation(self, humid, temp):
        #  온도에 대해서만 차이계산
        if humid > 0:
            carrot = 1.0 - 0.5 * abs(18.0 - temp) / 60
        #  전체 차이계산
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

class Brain:

    def __init__(self):
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.optimizer = None
        self.Q = None
        self.target_Q = None
        self.epsilon = None

    def modeling_NN(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(self.num_states, 32))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(32, 32))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc2', nn.Linear(32, self.num_actions))
        return model

    def modeling_OPTIM(self):
        optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=LEARNING_RATE)
        return optimizer

    def update_Q(self):
        data = agent.db.sampling(BATCH_SIZE)
        batch = DATA(*zip(*data))
        state_serial = batch.state
        action_serial = torch.cat(batch.action)
        reward_serial = torch.cat(batch.reward)
        next_state_serial = batch.next_state
        done_serial = batch.done

        update_input = np.zeros((BATCH_SIZE, self.num_states))
        update_target = np.zeros((BATCH_SIZE, self.num_states))
        action, reward, done = [], [], []

        for i in range(BATCH_SIZE):
            # state
            update_input[i] = state_serial[i]
            # action
            action.append(action_serial[i])
            # reward
            reward.append(reward_serial[i])
            # next_state
            update_target[i] = next_state_serial[i]
            # done
            done.append(done_serial[i])

        # 텐서형 통일
        update_input = torch.from_numpy(update_input)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        update_target = torch.from_numpy(update_target)
        done = torch.tensor(done)

        # Float형 통일 => 신경망 결과추출(y and y_hat)
        Q_val = self.Q(update_input.float())
        Target_Q_val = self.target_Q(update_target.float())

        for i in range(BATCH_SIZE):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                Q_val[i][action[i]] = reward[i]
            else:
                Q_val[i][action[i]] = reward[i] + DISCOUNT_FACTOR * max(Target_Q_val[i])

        #훈련 모드
        self.Q.train()
        # 손실함수 계산
        loss = F.smooth_l1_loss(update_input, Q_val)
        # 가중치 수정
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        exit()

    def update_Target_Q(self):
        self.target_Q = self.Q

    def action_order(self,state,episode):
        self.epsilon = 0.5 * (1 / (episode + 1))
        if self.epsilon <= np.random.uniform(0, 1):
            '''For Exploitation-이용'''
            self.Q.eval()
            with torch.no_grad():
                data = self.Q(state)
                action = torch.argmax(data).item()
        else:
            '''For Exploration-탐험'''
            action = random.randrange(self.num_actions)
        action = torch.tensor(action).view(1,1)
        action = action.squeeze(1)
        return action

class DB:

    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def save_to_DB(self, state, action, reward, next_state, done):

        if(len(self.memory) < self.capacity):
            self.memory.append(None)

        self.memory[self.index] = DATA(state, action, reward, next_state,done)
        self.index = (self.index+1) % self.capacity

    def sampling(self, batch_size):
        return random.sample(agent.db.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:# 마무리

    def __init__(self):
        self.brain = Brain()
        self.brain.Q = self.brain.modeling_NN()
        self.brain.optimizer = self.brain.modeling_OPTIM()
        self.brain.target_Q = self.brain.modeling_NN()
        self.db = DB()

    def update_Q_process(self):
        if self.db.__len__() < BATCH_SIZE:
            return
        else:
            self.brain.update_Q()

    def update_Target_Q_process(self):
        self.brain.update_Target_Q()

    def action_process(self,state,episode):
        return self.brain.action_order(state,episode)

    def save_process(self,state,action,reward,next_state,done):
        self.db.save_to_DB(state,action,reward,next_state,done)



if __name__ == '__main__':
    env = Carrot_House()
    agent = Agent()
    scores, episodes = [], []
    for E in range(EPISODES):
        state = env.reset()
        score = 0
        for S in range(MAX_STEPS):
            action = agent.action_process(state,E)
            next_state, reward, done = env.step(action)
            agent.save_process(state,action,reward,next_state,done)
            agent.update_Q_process()
            if done:
                break
            else:
                score += reward
                state = next_state
        agent.update_Target_Q_process()
        scores.append(score)
        episodes.append(E)
        print("episode:", E, "  score:", score, "  memory length:", len(agent.db.memory), "  epsilon:", agent.brain.epsilon)