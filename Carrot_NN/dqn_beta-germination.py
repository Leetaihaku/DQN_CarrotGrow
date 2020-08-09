import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
import copy

NUM_STATES = 2
NUM_ACTIONS = 4
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 256
NODES = 24
TRAIN_START = 1000
CAPACITY = 10000
EPISODES = 10000
MAX_STEPS = 200
EPSILON = 1.0
EPSILON_DISCOUNT_FACTOR = 0.0001
EPSILON_MIN = 0.01
PATH = '.\saved_model\Beta-17.pth'
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
        model.add_module('fc3', nn.Linear(NODES, self.num_actions))
        return model

    def modeling_OPTIM(self):
        optimizer = optim.RMSprop(self.Q.parameters(), lr=LEARNING_RATE)
        return optimizer

    def update_Q(self):
        data = agent.db.sampling(BATCH_SIZE)
        batch = DATA(*zip(*data))
        state_serial = batch.state
        action_serial = torch.cat(batch.action).reshape(-1, 1)
        reward_serial = torch.cat(batch.reward)
        next_state_serial = batch.next_state
        done_serial = batch.done

        state_serial = torch.stack(state_serial)
        next_state_serial = torch.stack(next_state_serial)
        done_serial = torch.tensor(done_serial)

        # Float형 통일 => 신경망 결과추출(y and y_hat)
        Q_val = self.Q(state_serial)
        Q_val = Q_val.gather(1, action_serial)
        Target_Q_val = self.target_Q(next_state_serial).max(1)[0]
        Target_Q_val = reward_serial + DISCOUNT_FACTOR * (~done_serial) * Target_Q_val
        Target_Q_val = Target_Q_val.reshape(-1, 1)

        # 훈련 모드
        self.Q.train()
        # 손실함수 계산
        loss = F.smooth_l1_loss(Target_Q_val, Q_val)
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
        '''Exploitation-이용'''
        self.Q.eval()
        with torch.no_grad():
            data = self.Q(state)
            action = torch.argmax(data).item()
        return action



class Agent:

    def __init__(self):
        self.brain = Brain()
        self.brain.Q = self.brain.modeling_NN()
        self.brain.target_Q = self.brain.modeling_NN()
        self.brain.optimizer = self.brain.modeling_OPTIM()
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



class Carrot_House:

    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Humid = 0
        self.Temp = 0
        self.Pre_temp = 0
        self.Cumulative = 0

    def Supply_water(self):
        self.Humid += 7

    def Temp_up(self):
        self.Temp += 1

    def Temp_down(self):
        self.Temp -= 1

    def Wait(self):
        return

    def Pre_step(self):
        # 스텝
        self.Cumulative += 1
        # 직전온도
        self.Pre_temp = self.Temp
        # 수분량 감소, 온도 변동
        if self.Humid > 0:
            self.Humid -= 1
        else:
            self.Humid = 0
        # self.Temp += random.randint(-1, 1)

    def Step(self, action):
        '''행동진행 => 환경결과'''
        #스텝환경 셋팅
        self.Pre_step()

        # 물주기
        if action == 0:
            self.Supply_water()
        # 온도 올리기
        elif action == 1:
            self.Temp_up()
        # 온도 내리기
        elif action == 2:
            self.Temp_down()
        # 현상유지
        elif action == 3:
            self.Wait()

        # 보상
        reward = self.Get_reward()

        # 종료조건
        if reward == -1:
            done = True
        elif self.Cumulative == MAX_STEPS:
            done = True
        else:
            done = False

        next_state = np.array([self.Humid, self.Temp])
        next_state = torch.from_numpy(next_state)
        next_state = next_state.float()
        reward = np.array([reward])
        reward = torch.from_numpy(reward)
        reward = reward.float()

        return next_state, reward, done

    def Get_reward(self):

        if self.Humid > 0 and self.Humid <= 7:
            if self.Temp <= 0:
                reward = -0.5
            elif abs(18.0 - self.Temp) < abs(18.0 - self.Pre_temp):
                reward = 0.5
            elif abs(18.0 - self.Temp) == abs(18.0 - self.Pre_temp) and self.Temp == 18.0:
                reward = 1
            elif abs(18.0 - self.Temp) > abs(18.0 - self.Pre_temp):
                reward = -0.5
            elif self.Humid == 7:
                reward = 1
            elif abs(18.0 - self.Temp) == abs(18.0 - self.Pre_temp) and self.Temp != 18.0:
                reward = -0.5
            else:
                reward = 0.0
        else:
            reward = -1

        return reward

    def reset(self):
        '''환경 초기화'''
        init_humid = np.random.randint(low=0, high=7)
        init_temp = np.random.randint(low=0, high=36)
        self.Humid = init_humid
        self.Temp = init_temp
        init_state = np.array([init_humid, init_temp])
        init_state = torch.from_numpy(init_state)
        init_state = torch.squeeze(init_state)

        # Humid, temp
        return init_state.float()


if __name__ == '__main__':
    env = Carrot_House()
    agent = Agent()
    agent.brain.Q.load_state_dict(torch.load(PATH))
    agent.brain.Q.eval()
    scores, episodes = [], []
    for E in range(EPISODES):
        state = env.reset()
        score = 0
        for S in range(MAX_STEPS):
            print(S)
            action = agent.action_process(state, E)
            next_state, reward, done = env.Step(action)
            if done:
                print("step", S, "  episode:", E,
                      "  score:", score, "  memory length:", len(agent.db.memory), "  epsilon:", agent.brain.epsilon)
                break
            else:
                score += reward
                state = next_state
        scores.append(score)
        episodes.append(E)
    print('Task End')