from Network import CriticNet
from Hyparameter import sampling_size
from PSR import PSR
from Environment import POMDPEnvironment
import numpy as np
from LogUtil import logger


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# agent module for learning
class agent(object):
    def __init__(self, action_space, length_or):
        self.action_space = action_space
        self.epsilon = 0.5
        self.initial_epsilon = 0.5
        self.final_epsilon = 0.05
        self.explore = 500
        self._memory = []
        self.max_memory = 2000 # number of previous transitions to remember
        self.observation_id = None
        self.sampling_size = sampling_size
        self.count = 0
        self.length_or = length_or

    # set the dimensionality of predictive state and action space and state space and discount rate for neural network
    def set_state_dim(self, state_dim, action_space, state_space, discount_rate):
        self.Net = CriticNet(state_dim=state_dim, action_space=action_space, state_space=state_space, discount_rate=discount_rate, length_or=self.length_or)

    # return the action_index it will act
    def taking_action(self, Predictive_State):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(low=0, high=len(self.action_space), size=1, dtype=np.int)
        else:
            action_index, Optimal_pro_z, all_Dis = self.Net.selecting_optimal_action(Predictive_State=Predictive_State, net='origin')
            # print('under current predictive state:', Predictive_State)
            # for i in range(len(all_Dis[0])):
            #     print('for action id:' + str(i), 'expectation:'+str(all_Dis[0][i]))
        if action_index.shape == (1,):
            action_index = action_index[0]
        else:
            print('exception on taking action function')
        return action_index

    def train_agent(self, PSR):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        if self.sampling_size > len(self._memory):
            _index = np.random.randint(low=0, high=len(self._memory), size=len(self._memory), dtype=np.int)
        else:
            _index = np.random.randint(low=0, high=len(self._memory), size=self.sampling_size, dtype=np.int)
        samples = np.array(self._memory)[_index]
        length = len(samples)
        #######################################################################################
        #try all possible actions
        _length_a = len(PSR.Actions)
        samples = list(samples) * _length_a
        _samples = None
        for i in range(length):
            a = samples[i:len(samples):length]
            if _samples is None:
                _samples = a
            else:
                _samples = np.concatenate([_samples, a], axis=0)
        _action = np.array(list(np.reshape(a=np.arange(0, _length_a, 1, np.int), newshape=(-1,))) * length)
        _samples[:, 1] = _action
        samples = _samples[:, :]
        #######################################################################################
        length = len(samples)
        _length_or = (len(PSR.Observations) * len(PSR.R_list))
        samples = list(samples) * _length_or
        _samples = None
        for i in range(length):
            a = samples[i:len(samples):length]
            if _samples is None:
                _samples = a
            else:
                _samples = np.concatenate([_samples, a], axis=0)
        length_o = len(PSR.Observations)
        length_r = len(PSR.R_list)
        o = np.array(list(np.reshape(a=np.arange(0, length_o, 1, np.int), newshape=(-1, 1))) * length_r)
        r = np.array(list(np.reshape(a=np.arange(0, length_r, 1, np.int), newshape=(-1, 1))) * length_o)
        _o = None
        for i in range(length_o):
            a = o[i:len(o):length_o]
            if _o is not None:
                _o = np.concatenate([_o, a], axis=0)
            else:
                _o = a
        _or = np.concatenate([_o, r], axis=-1)
        _or = np.array(list(_or)*length)
        _samples = np.array(_samples)
        samples = np.concatenate([_samples, _or], axis=1)

        a_id = samples[:, 1]
        o_id = samples[:, 2]
        r_id = samples[:, 3]
        R_id = np.asarray(r_id, np.int32)
        R = np.array(PSR.R_list)[R_id]
        R = np.reshape(R, (-1, 1))
        samples = np.concatenate([samples, R], axis=1)
        Predictive_State = samples[:, 0]
        Next_Predictive_State, Next_Predictive_State_Probability = PSR.update_batch(a_id=a_id, o_id=o_id, r_id=r_id, Predictive_State=Predictive_State)
        #######################################################################################
        Next_Predictive_State_Probability = np.reshape(a=Next_Predictive_State_Probability, newshape=(-1, 1))
        Next_Predictive_State = np.reshape(a=Next_Predictive_State, newshape=(-1, np.shape(PSR.b_h)[1]))
        samples = np.concatenate([samples, Next_Predictive_State_Probability, Next_Predictive_State], axis=1)

        self.Net._train(samples=samples)
        self.count = self.count + 1
        if self.count % 2 == 0:
            self.Net.target_train()
            self.Net.origin_Net.save_weights(filepath='origin_net_Mountain_Car.h5', overwrite=True)
            self.Net.target_Net.save_weights(filepath='target_net_Mountain_Car.h5', overwrite=True)

    def randomly_sample_an_predictive_state(self):
        index = np.random.randint(low=0, high=len(self._memory), size=1)
        return self._memory[index[0]][0]

    # storing the action and relative information in memory for learning
    def replay_memory(self, Last_Predictive_State):
        self._memory.append((Last_Predictive_State, 0))
        if len(self._memory) > self.max_memory:
            self._memory.pop(-1)
# Environment Module for responding actions from agent
# class Environment(object):
#     def __init__(self, R, Observation, O, State_Sapce, Transition_Matrix):
#         self.tiger_state_index = np.random.randint(low=0, high=len(State_Sapce), size=1, dtype=np.int)[0]
#         self.R = R
#         self.OBSERVATION = Observation
#         self.O = O
#         self.State_Space = State_Sapce
#         self.Transition_Matrix = Transition_Matrix
#
#     # tiger will randomly move after an agent acts open-left or open-right
#     def tiger_shift(self, action_idx):
#         state = np.arange(0, len(self.State_Space), 1, np.int32)
#         T = self.Transition_Matrix[action_idx][self.tiger_state_index]
#         new_state = np.random.choice(a=state, size=1, p=T)
#         self.tiger_state_index = new_state[0]
#         return self.tiger_state_index
#
#     # rendering the corresponding reward
#     def obtain_reward(self, agent_action_index, current_state, next_state, observation):
#         R = self.R[(agent_action_index, current_state, next_state, observation)]
#         return R
#
#     # receiving the action taken by an agent and transit back the observation and reward
#     def receive_action(self, action_idx):
#         current_state = self.tiger_state_index
#         next_state = current_state
#         if action_idx == 2:
#             next_state = self.tiger_shift(action_idx=action_idx)
#             if next_state != current_state:
#                 print('error on tiger shift!')
#             print('the tiger actually is in s'+str(current_state))
#         else:
#             next_state = self.tiger_shift(action_idx=action_idx)
#         o_list = []
#         for i in range(len(self.OBSERVATION)):
#             o_list.append(self.O[(action_idx, next_state, i)])
#         o_id_list = np.arange(0, len(self.OBSERVATION), 1, dtype=np.int)
#         o_list = np.array(o_list)
#         o_id = np.random.choice(a=o_id_list, size=1, p=o_list)
#         print('the observation probability:', o_list)
#         reward = self.obtain_reward(action_idx, current_state, next_state, o_id[0])
#         return [reward, o_id[0]]

#this environment is driven by PSR

class Environment(object):
    def __init__(self, Predictive_State, State_Space, Observation_Space, Reward_Space, m_ao, m_name):
        self.Predictive_State = Predictive_State
        self.S_Space = State_Space
        self.O_Space = Observation_Space
        self.R_Space = Reward_Space
        self.m_name = m_name
        self.m_ao = m_ao

    def get_all_possible_or(self, a_id):
        length_o = len(self.O_Space)
        length_r = len(self.R_Space)
        o = np.array(list(np.reshape(a=np.arange(0, length_o, 1, np.int), newshape=(-1, 1))) * length_r)
        r = np.array(list(np.reshape(a=np.arange(0, length_r, 1, np.int), newshape=(-1, 1))) * length_o)
        _o = None
        for i in range(length_o):
            a = o[i:len(o):length_o]
            if _o is not None:
                _o = np.concatenate([_o, a], axis=0)
            else:
                _o = a
        _a = np.ones(shape=(len(_o), 1), dtype=np.int) * a_id
        return [_a, _o, r]

    def Update_Predictive_State(self, Predictive_State):
        self.Predictive_State = Predictive_State

    def receive_action(self, action_idx):
        a_id_int, o_id_int, r_id_int = self.get_all_possible_or(a_id=action_idx)
        a_id = np.array(a_id_int, dtype=np.str)
        o_id = np.array(o_id_int, dtype=np.str)
        r_id = np.array(r_id_int, dtype=np.str)
        a = np.broadcast_to(array='a', shape=np.shape(a_id))
        o = np.broadcast_to(array='o', shape=np.shape(o_id))
        r = np.broadcast_to(array='r', shape=np.shape(r_id))
        a_id = np.core.defchararray.add(a, a_id)
        o_id = np.core.defchararray.add(o, o_id)
        r_id = np.core.defchararray.add(r, r_id)
        test = np.core.defchararray.add(a_id, np.core.defchararray.add(o_id, r_id))
        index = np.searchsorted(self.m_name, test)
        m_ao = np.array(self.m_ao)[index]
        m_ao = np.reshape(a=m_ao, newshape=(-1, np.shape(self.m_ao[0])[0], np.shape(self.m_ao[0])[1]))
        P_aor = np.matmul(a=self.Predictive_State.T, b=m_ao)
        P_aor = np.reshape(a=P_aor, newshape=(-1, ))
        P_aor = np.round(a=P_aor, decimals=12)
        value = P_aor[P_aor < 0]
        if len(value) > 0:
            print('error on receive action')
            print('the predictive state is:', self.Predictive_State.T)
            tmp = P_aor[:]
            tmp[tmp < 0] = 0
            print('the probability of sum:', np.sum(a=tmp, axis=-1))
            P_aor[P_aor<0] = 0
        index = np.arange(start=0, stop=len(test), step=1, dtype=np.int)
        _aor_idx = np.random.choice(a=index, size=1, p=P_aor)
        _aor_idx = int(_aor_idx)
        R = self.R_Space[int(r_id_int[_aor_idx])]
        o = o_id_int[_aor_idx]
        return [R, o[0]]


if __name__ == "__main__":
    #####initialization########################################################
    ## dynamic loading an pomdp environment file and generate transition matrix T and observation matrix O with others
    EnvObject = POMDPEnvironment(filename='MountainCar.POMDP')
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
    length_or = len(Observations) * len(R_Matrix)
    ############################################################################
    TOTAL_REWARD = 0
    EPISODE_REWARD = 0
    maximum_episode = 40000
    episode_length = 30
    # initializing Agent Env and PSR three modules
    Agent = agent(action_space=Actions, length_or=length_or)
    PSR = PSR(T=T, O=O, b_h=b_h, Observations=Observations, Actions=Actions, R_Matrix = R_Matrix)
    testset = PSR.generate_tests()
    U_T = PSR.Computing_U_T()
    U_Q_Name, U_Q = PSR.generate_U_Q()
    PSR.Predictive_State = PSR.return_predictive_state()
    Agent.set_state_dim(state_dim=PSR.gain_core_tests_dim(), action_space=Actions, state_space=States, discount_rate=discount_rate)
    #Agent.Net.origin_Net.load_weights(filepath='origin_net_Mountain_Car_uniform_noise.h5')
    #Agent.Net.target_Net.load_weights(filepath='target_net_Mountain_Car_uniform_noise.h5')
    PSR.gain_m()
    PSR.gain_M_ao()
    #set an initial predictive state and starts learning process
    Current_Predictive_State = PSR.Predictive_State
    initial_Predictive_State = Current_Predictive_State
    print('predictive state:', Current_Predictive_State)
    Env = Environment(Predictive_State=Current_Predictive_State, State_Space=States, Observation_Space=Observations, Reward_Space=PSR.R_list, m_ao=PSR.m, m_name=PSR.m_name)

    ###################################################################
    #measurement
    E_Optimal_Actions = []
    E_Optimal_Actions_std = []
    Epoch_Open_Count = []
    Epoch_Open_Reward = []
    Open_Reward = 0
    Avg_R_open = []
    count_open = 0
    ###################################################################

    for j in range(maximum_episode):
        for i in range(episode_length):
            action_idx = Agent.taking_action(Predictive_State=Current_Predictive_State)
            reward, Agent.observation_id = Env.receive_action(action_idx=action_idx)
            EPISODE_REWARD += reward
            r_id = PSR.R_list.index(reward)
            Next_Predictive_State = PSR.update(action_idx=action_idx, observation_id=Agent.observation_id, r_id=r_id)

            ############################################################
            if action_idx == 1 or action_idx == 2:
                count_open = count_open + 1
                Open_Reward = Open_Reward + reward
            #############################################################

            Agent.replay_memory(Last_Predictive_State=Current_Predictive_State)
            if (i+1) % episode_length == 0:
                print('the i is:', i)
                Agent.train_agent(PSR=PSR)
                #############################################################
                if count_open != 0:
                    Avg_Reward = EPISODE_REWARD / np.float(count_open)
                    Avg_Real_Reward_Open = Open_Reward / np.float(count_open)
                Avg_R_open.append(Avg_Reward)
                Epoch_Open_Reward.append(Avg_Real_Reward_Open)
                Epoch_Open_Count.append(count_open)
                a_id, Optimal_pro_z, all_Dis = Agent.Net.selecting_optimal_action(
                    Predictive_State=initial_Predictive_State, net='origin')
                std = np.sqrt(np.sum(Optimal_pro_z[0] * np.square(Agent.Net.z - np.sum(Agent.Net.z*Optimal_pro_z[0]))))
                E = Agent.Net.z * Optimal_pro_z[0]
                E = np.sum(a=E, axis=-1)
                E_Optimal_Actions.append(E)
                E_Optimal_Actions_std.append(std)

                np.save(file='Avg_reward_open.npy', arr=Avg_R_open)
                np.save(file='E_Optimal_Actions.npy', arr=E_Optimal_Actions)
                np.save(file='E_Optimal_Actions_std.npy', arr=E_Optimal_Actions_std)
                np.save(file='Avg_real_reward_open.npy', arr=Epoch_Open_Reward)
                np.save(file='Epoch_Open_Count.npy', arr=Epoch_Open_Count)
                #############################################################
            Current_Predictive_State = Next_Predictive_State
            Env.Update_Predictive_State(Predictive_State=Current_Predictive_State)
        logger.info('this' + str(j) +'episode rewards is:'+str(EPISODE_REWARD))
        TOTAL_REWARD += EPISODE_REWARD
        EPISODE_REWARD = 0
        count_open = 0
        Open_Reward = 0
        Predictive_State = Agent.randomly_sample_an_predictive_state()
        PSR.Predictive_State = Predictive_State
        Env.Update_Predictive_State(Predictive_State=Predictive_State)
    logger.info('after 600 episode, the total reward is:'+str(TOTAL_REWARD))
    np.save(file='Avg_reward_open.npy', arr=Avg_R_open)
    np.save(file='E_Optimal_Actions.npy', arr=E_Optimal_Actions)
    np.save(file='E_Optimal_Actions_std.npy', arr=E_Optimal_Actions_std)
    np.save(file='Avg_real_reward_open.npy', arr=Epoch_Open_Reward)
    np.save(file='Epoch_Open_Count.npy', arr=Epoch_Open_Count)