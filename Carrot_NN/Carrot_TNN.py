import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random

'''
    1)Replay MM(DB) => 학습재료
    2)Brain(=Behavior) => 행동선택
    3)Agent => 행동주체
    4)Environment => 상태, 보상 수여자
'''

ENV = 'Carrot_NN(TextMode)'
GAMMA = 0.99
MAX_STEP = 200
NUM_EPISODES = 500
BATCH_SIZE = 32
CAPACITY = 10000
LEARNING_RATE = 0.01
NUM_STATES = 3
NUM_ACTIONS = 4
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



class ReplayMemory:#DB => 학습용데이터 저장소

    def __init__(self, Capacity):
        self.capacity = Capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        '''Transition(종합상태) -> 메모리 저장'''
        #메모리 수용가능잔량 있을 시,
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        #저장
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index+1) % self.capacity

    def sample(self, batch_size):
        '''BATCH_SIZE만큼 무작위데이터 Transition 추출'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        '''len함수 -> 현재 저장 transition개수 파악'''
        return len(self.memory)



class Brain:#행동 결정자

    def __init__(self, num_states, num_actions):
        '''신경망 모델구성'''
        #행동의 가짓 수
        self.num_actions = num_actions
        #Trainsition 메모리 객체 생성
        self.memory = ReplayMemory(CAPACITY)
        #신경망 구성
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))
        #신경망 구성 출력
        print(self.model)
        #최적화 기법 선택
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)

    def replay(self):
        '''Replay Memory활용 학습'''
        print('DB길이 : ', len(self.memory))
        print('DB정보 : ', self.memory)
        
        if len(self.memory) < BATCH_SIZE:
            '''메모리 < 배치사이즈 -> 학습 없음'''
            return

        else:
            '''메모리 > 배치사이즈 -> 학습진행'''

            '''미니배치 생성'''
            #메모리 -> 미니배치 추출
            trainsition = self.memory.sample(BATCH_SIZE)
            #전처리1 -> 학습용 데이터변환<컬럼별 정리> :: (state*BATCH, ...)
            batch = Transition(*zip(*trainsition))
            #전처리2 -> 텐서 * 1 * 4 => 텐서 * 4
            print(batch.state)
            print(batch.action)
            print(batch.reward)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
            '''실측 Q 계산'''
            #추론모드 전환
            self.model.eval()
        
            #Q함수 계산[신경망 결과 = 행동(0:물주기 1:온도올리기 2:온도내리기)인덱스 및 Q값 구하기]
            print(state_batch)
            state_action_values = self.model(state_batch).gather(1, action_batch)
            #Done or Not => Masking
            non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
            #전체 초기화??
            next_state_values = torch.zeros(BATCH_SIZE)
            #각 상태의 최대Q행동[Value, Index] -> Value 추출
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

            #★★★ Q러닝식 => 실측값(에러계산)사용 ★★★
            expected_state_action_values = reward_batch + GAMMA * next_state_values

            '''가중치 수정'''
            #학습모드 전환
            self.model.train()

            #손실함수 계산
            loss = F.smooth_l1_loss(state_action_values,expected_state_action_values)
            #가중치 수정
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def decide_action(self, state, episode):
        '''상태기반 행동결정'''
        #E-greedy choice
        epsilon = 0.5 * (1/(episode+1))

        if epsilon <= np.random.uniform(0, 1):
            print('이용 조건 진입')
            '''For Exploitation-이용'''
            #추론모드 전환
            self.model.eval()
            #텐서변환 => Torch Lonetensor -> size 1*1
            with torch.no_grad():
                #init_state = torch.argmax(init_state)
                print(state)
                data = self.model(state)
                action = torch.argmax(data).view(1, 1)

        else:
            print('탐험 조건 진입')
            '''For Exploration-탐험'''
            #학습모드 유지
            #행동 무작위 반환
            #텐서변환 필요없음
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action



class Agent:#행동자

    def __init__(self, num_states, num_actions):
        '''상태 및 행동 가짓수 결정'''
        self.brain = Brain(num_states, num_actions)

    def update_Q(self):
        '''Q함수 수정'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''행동'''
        action = self.brain.decide_action(state, episode)
        return action
    
    def memorize(self, state, action, next_state, reward):
        '''DB저장'''
        self.brain.memory.push(state, action, next_state, reward)



class Environment:#환경

    def __init__(self):
        self.env = Carrot_House()
        
    def run(self):
        '''실행'''
        print('RUN함수 실행')
        for episode in range(NUM_EPISODES):
            '''전체 에피소드 반복문'''
            print('{} 에피소드 시작'.format(episode))
            #state = 당근 상태
            state = self.env.reset()
            print('리셋 성공')
            
            for step in range(MAX_STEP):
                print('{}번째 스텝'.format(step))
                '''에피소드 진행 반복문'''
                print('에피소드 루프진입')
                #행동 결정
                action = self.env.agent.get_action(state, episode)
                print('행동 함수성공')
                #다음 상태, 보상, 종료여부
                next_state, reward, done = self.env.step(action.item())
                print('다음상태', next_state, ', 보상', reward, ', 종료여부함수', done)

                #에피소드 종료 or 다음스텝
                if done:
                    print('종료 ㅇ 조건 진입')
                    if state[0] > 0.9:
                        reward += 1.0
                        print("당근 재배 성공")
                        break
                    else:
                        reward -= 1.0
                        print("당근 재배 실패")
                        break
                else:
                    print('진행 조건 진입')
                    reward -= 0.001

                #메모리 저장(보상 텐서화)
                reward = torch.FloatTensor([reward])
                self.env.agent.memorize(state, action, next_state, reward)
                print('메모리 저장 성공')

                #Q함수 업데이트
                self.env.agent.update_Q()
                print('Q업데이트 성공')

                #다음상태로 넘어가기
                state = next_state
                print('---상태전환 스텝 마무리---')

class Carrot_House:#하우스 환경
    
    def __init__(self):
        '''하우스 환경 셋팅'''
        self.Carrot = 0.0
        self.Humid = 0
        self.Temp = 0.0
        self.Cumulative_Step = 0
        self.agent = Agent(NUM_STATES, NUM_ACTIONS)

    def step(self, action):
        '''행동진행 => 환경결과'''
        #물주기
        if action == 0:
            self.supply_water()
        #온도 올리기
        elif action == 1:
            self.Temp_up()
        #온도 내리기
        elif action == 2:
            self.Temp_down()
        #현상유지
        elif action == 3:
            self.Wait()

        self.Carrot = self.HP_calculation(self.Humid, self.Temp)

        #하루치 수분량 감쇄
        self.Humid -= 1

        #종료여부
        if self.Cumulative_Step == MAX_STEP:
            done = True
        else:
            done = False

        # 보상
        reward = -0.001

        next_state = np.array([self.Carrot, self.Humid, self.Temp])
        next_state = torch.from_numpy(next_state)
        next_state = next_state.float()
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
        #온도에 대해서만 차이계산
        if humid > 0:
            carrot = 1.0 - 0.5 * abs(18.0 - temp) / 60
        #전체 차이계산
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

if __name__ == '__main__':
    Carrot_env = Environment()
    Carrot_env.run()
