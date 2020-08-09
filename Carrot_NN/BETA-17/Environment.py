import torch
import numpy as np
from Hyperparams import MAX_STEPS



class Carrot_House:

    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Humid = 0
        self.Temp = 0
        self.Pre_temp = 0
        self.Cumulative = 0

    def supply_water(self):
        self.Humid += 7

    def temp_up(self):
        self.Temp += 1

    def temp_down(self):
        self.Temp -= 1

    def wait(self):
        return

    def pre_step(self):
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

    def step(self, action):
        '''행동진행 => 환경결과'''

        #스텝환경 셋팅
        self.pre_step()

        # 물주기
        if action == 0:
            self.supply_water()
        # 온도 올리기
        elif action == 1:
            self.temp_up()
        # 온도 내리기
        elif action == 2:
            self.temp_down()
        # 현상유지
        elif action == 3:
            self.wait()

        # 보상
        reward = self.get_reward()

        # 종료조건
        if reward == -1:
            done = True
        elif self.Cumulative == MAX_STEPS:
            done = True
        else:
            done = False

        next_state = torch.tensor([self.Humid, self.Temp]).float()
        reward = torch.tensor([reward]).float()

        return next_state, reward, done

    def get_reward(self):

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
        init_state = torch.tensor([init_humid, init_temp])

        return init_state.float()