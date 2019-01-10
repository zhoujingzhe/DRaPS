from Network import CriticNet
from Hyparameter import sampling_size
from PSR import PSR
from Environment import POMDPEnvironment
import numpy as np
from LogUtil import logger

# agent module for learning
class agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 0.5
        self.initial_epsilon = 0.5
        self.final_epsilon = np.finfo(float).eps
        self.explore = 1000
        self.memory_a0 = []
        self.memory_a1 = []
        self.memory_a2 = []
        self.max_memory = 2000 # number of previous transitions to remember
        self.observation_id = None
        self.sampling_size = sampling_size

    # set the dimensionality of predictive state and action space and state space and discount rate for neural network
    def set_state_dim(self, state_dim, action_space, state_space, discount_rate):
        self.Net = CriticNet(state_dim=state_dim, action_space=action_space, state_space=state_space, discount_rate=discount_rate)

    # return the action_index it will act
    def taking_action(self, Predictive_State):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(low=0, high=len(self.action_space), size=1, dtype=np.int)
        else:
            action_index, Optimal_pro_z, all_Dis = self.Net.selecting_optimal_action(Predictive_State=Predictive_State, net='origin')
            print('under current predictive state:', Predictive_State)
            for i in range(len(all_Dis[0])):
                print('for action id:' + str(i), 'expectation:'+str(all_Dis[0][i]))
        if action_index.shape == (1,):
            action_index = action_index[0]
        else:
            print('exception on taking action function')
        return action_index

    # storing the action and relative information in memory for learning
    def replay_memory(self, Last_Predictive_State, action_idx, r_t, Next_Predictive_State):
        if action_idx == 0:
            self.memory_a0.append((Last_Predictive_State, action_idx, r_t, Next_Predictive_State))
        elif action_idx == 1:
            self.memory_a1.append((Last_Predictive_State, action_idx, r_t, Next_Predictive_State))
        elif action_idx == 2:
            self.memory_a2.append((Last_Predictive_State, action_idx, r_t, Next_Predictive_State))

        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory_a0) > self.max_memory:
            self.memory_a0.pop(-1)
        if len(self.memory_a1) > self.max_memory:
            self.memory_a1.pop(-1)
        if len(self.memory_a2) > self.max_memory:
            self.memory_a2.pop(-1)

        sampling_length = len(self.memory_a0) + len(self.memory_a1) + len(self.memory_a2)
        if sampling_length % self.sampling_size == 0 or sampling_length > 100:
            size = int(self.sampling_size / len(self.action_space))
            if len(self.memory_a0) < size:
                index_a0 = np.random.randint(0, len(self.memory_a0), size=len(self.memory_a0), dtype=np.int)
            else:
                index_a0 = np.random.randint(0, len(self.memory_a0), size=size,
                                              dtype=np.int)
            if len(self.memory_a1) < size:
                index_a1 = np.random.randint(0, len(self.memory_a1), size=len(self.memory_a1), dtype=np.int)
            else:
                index_a1 = np.random.randint(0, len(self.memory_a1), size=size,
                                              dtype=np.int)
            if len(self.memory_a2) < size:
                index_a2 = np.random.randint(0, len(self.memory_a2), size=len(self.memory_a2), dtype=np.int)
            else:
                index_a2 = np.random.randint(0, len(self.memory_a2), size=size,
                                              dtype=np.int)
            sample_a0 = np.array(self.memory_a0)[index_a0]
            sample_a1 = np.array(self.memory_a1)[index_a1]
            sample_a2 = np.array(self.memory_a2)[index_a2]
            samples = np.concatenate([sample_a0, sample_a1, sample_a2], axis=0)
            self.Net._train(samples=samples)
            self.Net.target_train()
            self.Net.origin_Net.save_weights(filepath='origin_net.h5', overwrite=True)
            self.Net.target_Net.save_weights(filepath='target_net.h5', overwrite=True)


# Environment Module for responding actions from agent
class Environment(object):
    def __init__(self, R, Observation, O):
        self.tiger_state_index = np.random.randint(low=0, high=2, size=1, dtype=np.int)[0]
        self.R = R
        self.OBSERVATION = Observation
        self.O = O

    # tiger will randomly move after an agent acts open-left or open-right
    def tiger_shift(self):
        action = np.random.choice(a=['move_left', 'move_right'], size=1, p=[0.5, 0.5])
        if action == 'move_left':
            self.tiger_state_index -= 1
            self.tiger_state_index = max(self.tiger_state_index, 0)
        elif action == 'move_right':
            self.tiger_state_index += 1
            self.tiger_state_index = min(self.tiger_state_index, 1)
        return self.tiger_state_index

    # rendering the corresponding reward
    def obtain_reward(self, agent_action_index, current_state, next_state, observation):
        R = self.R[(agent_action_index, current_state, next_state, observation)]
        return R

    # receiving the action taken by an agent and transit back the observation and reward
    def receive_action(self, action_idx):
        current_state = self.tiger_state_index
        next_state = None
        if action_idx == 2:
            next_state = current_state
            print('the tiger actually is in s'+str(current_state))
        elif action_idx == 0 or action_idx == 1:
            next_state = self.tiger_shift()
        else:
            print('exception on receive action!')
        o_list = []
        for i in range(len(self.OBSERVATION)):
            o_list.append(self.O[(action_idx, next_state, i)])
        o_id_list = np.arange(0, len(self.OBSERVATION), 1, dtype=np.int)
        o_id = np.random.choice(a=o_id_list, size=1, p=o_list)
        print('the observation probability:', o_list)
        reward = self.obtain_reward(action_idx, current_state, next_state, o_id[0])
        return [reward, o_id[0]]

if __name__ == "__main__":
    #####initialization########################################################
    ## dynamic loading an pomdp environment file and generate transition matrix T and observation matrix O with others
    EnvObject = POMDPEnvironment(filename='tiger.95.POMDP')
    T = EnvObject._obtain_transition()
    O = EnvObject._obtain_observation()
    Z = EnvObject.Z
    R_Matrix = EnvObject._obtain_reward_matrix()
    b_h = EnvObject._obtain_b_h()
    b_h = np.reshape(a=b_h, newshape=(1, -1))
    discount_rate = EnvObject.discount
    Observations = EnvObject.observations
    States = EnvObject.states
    Actions = EnvObject.actions
    ############################################################################
    TOTAL_REWARD = 0
    EPISODE_REWARD = 0
    maximum_episode = 600
    episode_length = 30
    # initializing Agent Env and PSR three modules
    Agent = agent(action_space=Actions)
    Env = Environment(R=EnvObject._obtain_rewards(), Observation=Observations, O=Z)
    PSR = PSR(T=T, O=O, b_h=b_h, Observations=Observations, Actions=Actions, R_Matrix = R_Matrix)
    testset = PSR.generate_tests()
    U_T = PSR.Computing_U_T()
    U_Q_Name, U_Q = PSR.generate_U_Q()
    PSR.Predictive_State = PSR.return_predictive_state()
    Agent.set_state_dim(state_dim=PSR.gain_core_tests_dim(), action_space=Actions, state_space=States, discount_rate=discount_rate)
#    Agent.Net.origin_Net.load_weights(filepath='origin_net.h5')
#    Agent.Net.target_Net.load_weights(filepath='target_net.h5')

    #set an initial predictive state and starts learning process
    Current_Predictive_State = PSR.Predictive_State
    print('predictive state:', Current_Predictive_State)
    for j in range(maximum_episode):
        for i in range(episode_length):
            action_idx = Agent.taking_action(Predictive_State=Current_Predictive_State)
            reward, Agent.observation_id = Env.receive_action(action_idx=action_idx)
            EPISODE_REWARD += reward
            r_id = PSR.R_list.index(reward)
            Next_Predictive_State = PSR.update(action_idx=action_idx, observation_id=Agent.observation_id, r_id=r_id, count=i)
            print('action id:', action_idx)
            print('observation id:', Agent.observation_id)
            print('reward id:', r_id)
            print('the new predictive state:', Next_Predictive_State)
            Agent.replay_memory(Last_Predictive_State=Current_Predictive_State, action_idx=action_idx, r_t=reward, Next_Predictive_State=Next_Predictive_State)
            Current_Predictive_State = Next_Predictive_State
        logger.info('this' + str(j) +'episode rewards is:'+str(EPISODE_REWARD))
        TOTAL_REWARD += EPISODE_REWARD
        EPISODE_REWARD = 0
        PSR.b_h = b_h
        Current_Predictive_State = PSR.return_predictive_state()
        PSR.Predictive_State = Current_Predictive_State
    logger.info('after 600 episode, the total reward is:'+str(TOTAL_REWARD))
#    from Visualizing import Visualizing_Distribution_On_ActionStatePair
#    Visualizing_Distribution_On_ActionStatePair(Agent=Agent, PSR=PSR, Env=Env)