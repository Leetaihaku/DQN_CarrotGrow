import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import copy
import random
import gym
from collections import namedtuple

QPATH = './saved_model/Validation_ISQ.pth'
TQPATH = './saved_model/Validation_ISTQ.pth'
DATA = namedtuple('DATA', ('state', 'action', 'reward', 'next_state', 'done'))

NUM_STATES = 4
NUM_ACTIONS = 2
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NODES = 24
TRAIN_START = 1
CAPACITY = 1000000
EPISODES = 18000
MAX_STEPS = 40000000
EPSILON = 1.0
EPSILON_DISCOUNT_FACTOR = 0.9999
EPSILON_MIN = 0.05


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
        #model.add_module('dropout1', nn.Dropout(p=0.5))
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        #model.add_module('dropout2', nn.Dropout(p=0.5))
        model.add_module('fc3', nn.Linear(NODES, self.num_actions))
        return model.cuda()

    def modeling_OPTIM(self):
        optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)
        return optimizer

    def update_Q(self):
        data = agent.db.sampling(BATCH_SIZE)
        batch = DATA(*zip(*data))
        state_serial = batch.state
        action_serial = torch.cat(batch.action)
        reward_serial = batch.reward
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
        update_input = torch.from_numpy(update_input).float().cuda()
        action = torch.tensor(action, device=gpu)
        reward = torch.tensor(reward, device=gpu)
        update_target = torch.from_numpy(update_target).float().cuda()

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

        # 훈련 모드
        self.Q.train()
        # 손실함수 계산
        loss = F.mse_loss(Q_origin, Q_val)
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
        # 최소 탐험성 확보
        #self.epsilon = EPSILON - EPSILON_DISCOUNT_DACTOR * episode if self.epsilon > EPSILON_MIN else EPSILON_MIN
        self.epsilon *= EPSILON_DISCOUNT_FACTOR
        state = torch.from_numpy(state).float().cuda()

        if self.epsilon < np.random.uniform(0, 1):
            '''For Exploitation-이용'''
            #print('이용')
            self.Q.eval()
            with torch.no_grad():
                data = self.Q(state)
                action = torch.argmax(data).item()
        else:
            '''For Exploration-탐험'''
            #print('탐험')
            action = random.randrange(self.num_actions)
        action = torch.tensor(action, device=gpu).view(1, 1)
        action = action.squeeze(1)
        return action


class Agent:  # 마무리

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


if __name__ == '__main__':
    gpu = torch.device('cuda')
    env = gym.make('CartPole-v0').unwrapped
    agent = Agent()
    agent.brain.Q.load_state_dict(torch.load(QPATH))
    agent.brain.target_Q.load_state_dict(torch.load(TQPATH))
    scores, episodes = [], []
    for E in range(EPISODES):
        state = env.reset()
        score = 0
        for S in range(MAX_STEPS):
            #env.render()
            action = agent.action_process(state, E)
            next_state, reward, done, info = env.step(action.item())
            agent.save_process(state, action, reward, next_state, done)
            if E >= TRAIN_START:
                agent.update_Q_process()
            torch.save(agent.brain.Q.state_dict(), QPATH)
            torch.save(agent.brain.target_Q.state_dict(), TQPATH)
            if done or S == MAX_STEPS-1:
                print("step", S, "  episode:", E,
                      "  score:", score, "  memory length:", len(agent.db.memory), "  epsilon:", agent.brain.epsilon)
                break
            else:
                score += reward
                state = next_state
        agent.update_Target_Q_process()
        scores.append(score)
        episodes.append(E)
        print('★★★★★★★★★★', E, '번째 학습 종료★★★★★★★★★★★')
    env.close()
    print('프로세스 종료')