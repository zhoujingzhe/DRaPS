from Network import CriticNet
from Hyparameter import batch_size, sampling_size, predictive_state_space
from PSR import PSR
from Environment import POMDPEnvironment
import numpy as np
from LogUtil import logger
import keras.backend as K
# agent module for learning
class agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.current_state_index = 0
        self.epsilon = 0.5
        self.initial_epsilon = 0.5
        self.final_epsilon = np.finfo(float).eps
        self.explore = 5000
        self.memory = []
        self.max_memory = 2000 # number of previous transitions to remember
        self.observation_id = None
        self.batch_size = batch_size
        self.sampling_size = sampling_size

    # set an predictive state index as current state
    def set_state(self, pred_state_index):
        self.current_state_index = pred_state_index

    # set the dimensionality of predictive state and action space and state space and discount rate for neural network
    def set_state_dim(self, state_dim, action_space, state_space, discount_rate):
        self.Net = CriticNet(state_dim=state_dim, action_space=action_space, state_space=state_space, discount_rate=discount_rate)

    # return the action_index it will act
    def taking_action(self):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(low=0, high=len(self.action_space), size=1, dtype=np.int)
        else:
            action_index, Optimal_pro_z = self.Net.selecting_optimal_action(index_state=[self.current_state_index], net='origin')
        if action_index.shape == (1,):
            action_index = action_index[0]
        else:
            print('exception on taking action function')
        return action_index

    # storing the action and relative information in memory for learning
    def replay_memory(self, s_index, action_idx, r_t, s1_index):
        self.memory.append((s_index, action_idx, r_t, s1_index))
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.pop(-1)

        sampling_length = len(self.memory)
        if sampling_length % 30 == 0:
            sampling_index = np.random.randint(low=0, high=sampling_length-1, size=self.sampling_size, dtype=np.int)
            samples = np.array(self.memory)[sampling_index]
            self.Net._train(samples=samples, batch_size=batch_size)
            self.Net.target_train()
            self.Net.origin_Net.save_weights(filepath='origin_net.h5', overwrite=True)
            self.Net.target_Net.save_weights(filepath='target_net.h5', overwrite=True)


# Environment Module for responding actions from agent
class Environment(object):
    def __init__(self, R, Observation):
        self.tiger_state_index = np.random.randint(low=0, high=2, size=1, dtype=np.int)[0]
        self.R = R
        self.OBSERVATION = Observation
    # tiger will randomly move after an agent acts open-left or open-right
    def tiger_shift(self):
        move_left = np.random.rand()
        move_right = 1 - move_left
        if move_left > move_right:
            self.tiger_state_index -= 1
            self.tiger_state_index = max(self.tiger_state_index, 0)
        elif move_left < move_right:
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
        next_state = self.tiger_shift()
        observation = None
        if action_idx != 2:
            if current_state == action_idx:
                observation = current_state
            else:
                observation = action_idx
        else:
            pro = np.random.rand()
            if pro < 0.15:
                observation = (current_state+1) % 2
            else:
                observation = current_state
        reward = self.obtain_reward(action_idx,current_state,next_state,observation)
        return [reward, observation]

if __name__ == "__main__":
    #####initialization########################################################
    ## dynamic loading an pomdp environment file and generate transition matrix T and observation matrix O with others
    EnvObject = POMDPEnvironment(filename='tiger.95.POMDP')
    T = EnvObject._obtain_transition()
    O = EnvObject._obtain_observation()
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
    #############################################################################


    ###############
    # initializing Agent Env and PSR three modules
    Agent = agent(action_space=Actions)
    Env = Environment(R=EnvObject.R, Observation=Observations)
    PSR = PSR(T=T, O=O, b_h=b_h, Observations=Observations, Actions=Actions)
    testset = PSR.generate_tests()
    U_T = PSR.Computing_U_T()
    U_Q_Name, U_Q = PSR.generate_U_Q()
    PSR.Predictive_State = PSR.return_predictive_state()
    Agent.set_state_dim(state_dim=PSR.gain_core_tests_dim(), action_space=Actions, state_space=States, discount_rate=discount_rate)
#    Agent.Net.origin_Net.load_weights(filepath='origin_net.h5')
#    Agent.Net.target_Net.load_weights(filepath='target_net.h5')

    #set an initial predictive state and starts learning process
    global predictive_state_space
    predictive_state_space.append(PSR.return_predictive_state())
    Agent.set_state(pred_state_index=0)
    for j in range(maximum_episode):
        for i in range(episode_length):
            action_idx = Agent.taking_action()
            reward, Agent.observation_id = Env.receive_action(action_idx =action_idx)
            EPISODE_REWARD += reward
            predictive_state = PSR.update(action_idx=action_idx, observation_id=Agent.observation_id, count=i)
            predictive_state_space.append(predictive_state)
            index = len(predictive_state_space) - 1
            Agent.replay_memory(s_index=index-1, action_idx=action_idx, r_t=reward, s1_index=index)
            Agent.set_state(pred_state_index=index)
            logger.info('Episode'+str(j)+'iteration'+str(i))
        logger.info('this' + str(j) +'episode rewards is:'+str(EPISODE_REWARD))
        TOTAL_REWARD += EPISODE_REWARD
    logger.info('after 600 episode, the total reward is:'+str(TOTAL_REWARD))