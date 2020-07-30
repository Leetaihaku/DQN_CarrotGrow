from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name="Carrot", seed=1, side_channels=[])

env.reset()
behavior_names = env.behavior_specs.keys()

env.reset()                 #리셋함수                 <반환>없음
env.step()                  #스텝함수                 <반환>없음
env.close()                 #통신단절함수              <반환>없음
env.behavior_specs()        #매핑함수(행동이름-행동)     <반환>없음
env.get_steps('''behavior_names: str(=입력행동이름)''')#<반환>행동이 요구되는 상태:결정튜플, 끝난상태:종결튜플반환
env.set_actions('''behavior_names: str(=입력행동이름), action: np.array''')#<반환>1차원 = 액션요구 주체, 2차원 = 선택가능행동 수
env.set_action_for_agent()#특정 에이전트만을 위한 행동

'''
    결정스텝들 - DecisionSteps -> 에이전트들의 전체배치정보
    결정스텝 - DecisionStep -> 한 에이전트의 배치정보
    
    결정스텝의 속성들
    1) obs          = 
    2) reward       = 
    3) agent_id     = 
    4) action_mask  = 
'''