import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
import copy

NUM_STATES = 3
NUM_ACTIONS = 4
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.1
BATCH_SIZE = 256
NODES = 32
TRAIN_START = 1000
CAPACITY = 1000
EPISODES = 15000
MAX_STEPS = 200
EPSILON = 1.0
EPSILON_DISCOUNT_DACTOR = 0.0001
EPSILON_MIN = 0.1
DATA = namedtuple('DATA', ('state', 'action', 'reward', 'next_state', 'done'))

class DB:

    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def save_to_DB(self, state, action, reward, next_state, done):
        if (len(self.memory) < self.capacity):
            self.memory.append(None)

        self.memory[self.index] = DATA(state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sampling(self, batch_size):
        return random.sample(agent.db.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Brain:

    def __init__(self):
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.optimizer = None
        self.Q = None
        self.target_Q = None
        self.epsilon = EPSILON

    def modeling_NN(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(self.num_states, NODES))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, self.num_actions))
        return model

    def modeling_OPTIM(self):
        optimizer = optim.RMSprop(self.Q.parameters(), lr=LEARNING_RATE)
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
        Q_origin = Q_val.clone().detach()

        for i in range(BATCH_SIZE):
            # Q러닝
            if done[i]:
                Q_val[i][action[i]] = reward[i]
            else:
                Q_val[i][action[i]] = reward[i] + DISCOUNT_FACTOR * max(Target_Q_val[i])

        #훈련 모드
        self.Q.train()
        # 손실함수 계산
        loss = F.smooth_l1_loss(Q_origin, Q_val)
        # 가중치 수정 프로세스
        # 옵티마이저 클리너
        self.optimizer.zero_grad()
        # 역전파 알고리즘
        loss.backward()
        # 가중치 수정
        self.optimizer.step()

    def update_Target_Q(self):
        self.target_Q = copy.deepcopy(self.Q)

    def action_order(self, state, episode):
        #최소 탐험성 확보
        self.epsilon = EPSILON - EPSILON_DISCOUNT_DACTOR * episode if self.epsilon > EPSILON_MIN else EPSILON_MIN

        if self.epsilon <= np.random.uniform(0, 1):
            '''For Exploitation-이용'''
            self.Q.eval()
            with torch.no_grad():
                data = self.Q(state)
                action = torch.argmax(data).item()
        else:
            '''For Exploration-탐험'''
            action = random.randrange(self.num_actions)
        action = torch.tensor(action).view(1, 1)
        action = action.squeeze(1)
        return action

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

    def action_process(self, state, episode):
        return self.brain.action_order(state, episode)

    def save_process(self, state, action, reward, next_state, done):
        self.db.save_to_DB(state, action, reward, next_state, done)

class Carrot_House  :  #  하우스 환경

    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Carrot = 0.0
        self.Humid = 0
        self.Temp = 0.0
        self.Cumulative_Step = 0

    def supply_water(self):
        self.Humid += 7

    def Temp_up(self):
        self.Temp += 5.0

    def Temp_down(self):
        self.Temp -= 5.0

    def Wait(self):
        return

    def Humid_calculation(self):
        if self.Humid <= 9 and self.Humid > 6:
            return 0.5
        elif self.Humid<= 6 and self.Humid > 4:
            return 0.25
        elif self.Humid <= 4 and self.Humid > 2:
            return 0.0
        elif self.Humid <= 2 and self.Humid > 0:
            return -0.25
        else:
            return -0.5

    def Temp_calculation(self, pre_temp):
        gap = abs(18.0 - self.Temp)
        pre_gap = abs(18.0 - pre_temp)

        # 온도 조정없이 오차범위 2도씨 이내
        if gap == pre_gap and gap <= 2:
            return 0.5
        # 온도 조정없이 오차범위 2도씨 이외(나태 패널티)
        if gap == pre_gap and gap > 2:
            return -0.25
        # 발전
        elif gap < pre_gap:
            return 0.25
        # 후퇴
        elif gap > pre_gap:
            return -0.5
        else:
            print('온도계산이상')

    def step(self, action):
        '''행동진행 => 환경결과'''
        #스텝변수++
        self.Cumulative_Step += 1
        #이전온도 => 보상계산에 사용
        pre_temp = self.Temp

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

        #당근체력
        self.Carrot = self.Humid_calculation() + self.Temp_calculation(pre_temp)

        #보상
        if self.Carrot >= 0.75:
            reward = 1
        elif self.Carrot >= 0.25 and self.Carrot < 0.75:
            reward = 0.5
        elif self.Carrot >= -0.25 and self.Carrot < 0.25:
            reward = 0
        elif self.Carrot >= -0.75 and self.Carrot < -0.25:
            reward = -0.5
        elif self.Carrot >= -1 and self.Carrot < -0.75:
            reward = -1
        else:
            print('보상할당오류', self.Carrot)
            
        #정류제재
        if self.Carrot < 0.5 and action == 3:
            reward -= 0.5

        #종료조건
        if reward <= -1:
            done = True
        else:
            done = False

        next_state = np.array([self.Carrot, self.Humid, self.Temp])
        next_state = torch.from_numpy(next_state)
        next_state = next_state.float()
        reward = np.array([reward])
        reward = torch.from_numpy(reward)
        reward = reward.float()

        # 수분량 감소, 온도 변동
        if self.Humid > 0:
            self.Humid -= 1
        else:
            self.Humid = 0
        #self.Temp += random.randint(-1, 1)

        return next_state, reward, done

    def reset(self):
        '''환경 초기화'''
        self.Carrot = 0.0
        self.Humid = 0
        self.Temp = 0.0
        self.Cumulative_Step = 0
        init_carrot = np.array([0.0])
        init_humid = np.array([0.0])
        init_temp = np.array([0.0])
        #init_humid = np.random.uniform(low=0.0, high=7.0, size=1)
        #init_temp = np.random.uniform(low=0.0, high=36.0, size=1)
        init_state = np.array([init_carrot, init_humid, init_temp])
        init_state = torch.from_numpy(init_state)
        init_state = torch.squeeze(init_state, 1)
        # Carrot, Humid, temp
        return init_state.float()

if __name__ == '__main__':
    env = Carrot_House()
    agent = Agent()
    scores, episodes = [], []
    for E in range(EPISODES):
        state = env.reset()
        score = 0
        for S in range(MAX_STEPS):
            print('step', S)
            print('state', state)
            action = agent.action_process(state, E)
            print('action', action)
            next_state, reward, done = env.step(action)
            agent.save_process(state, action, reward, next_state, done)
            if E > TRAIN_START:
                agent.update_Q_process()
            if done:
                print("final carrot_HP:", next_state[0], "  step", S, "  episode:", E,
                      "  score:", score, "  memory length:", len(agent.db.memory), "  epsilon:", agent.brain.epsilon)
                break
            elif S == MAX_STEPS-1:
                print('★★★성공한 훈련★★★')
            else:
                score += reward
                state = next_state
        agent.update_Target_Q_process()
        scores.append(score)
        episodes.append(E)
    print('학습 완료')