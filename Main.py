from Network import CriticNet
from Hyparameter import state_space, action_space, Open_Left, Open_Right, Listen, real_state_space, Tiger_Left, Tiger_Right, Evade_reward, Eaten_reward, Listen_reward, batch_size, sampling_size
from PSR import PSR
from keras.models import save_model

import numpy as np
class agent(object):
    def __init__(self):
        self.action_space = action_space
        self.current_state_index = 0
        self.epsilon = 0.99
        self.initial_epsilon = 0.99
        self.final_epsilon = np.finfo(float).eps
        self.explore = 5000
        self.memory = []
        self.max_memory = 2000 # number of previous transitions to remember
        self.tiger_observation = None
        self.batch_size = batch_size
        self.sampling_size = sampling_size

    def set_state(self, pred_state_index):
        self.current_state_index = pred_state_index

    def set_state_dim(self, state_dim):
        self.Net = CriticNet(state_dim=state_dim)

    #return the action_index it will act
    def taking_action(self):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(low=0, high=len(self.action_space), size=1, dtype=np.int)
        else:
            action_index, Optimal_pro_z = self.Net.selecting_optimal_action(index_state=self.current_state_index, net=self.Net.origin_Net, batch_size=1)
        if action_index.shape == (1,):
            action_index = action_index[0]
        else:
            print('exception on taking action function')
        return action_index

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

class Environment(object):
    def __init__(self):
        self.tiger_state_index = np.random.randint(low=0, high=2, size=1, dtype=np.int)[0]
        self.real_state_space = real_state_space
        self.action_space = action_space

    def tiger_shift(self):
        move_left = np.random.rand()
        move_right = 1 - move_left
        if move_left > move_right:
            self.tiger_state_index -= 1
            self.tiger_state_index = max(self.tiger_state_index, self.real_state_space.index(Tiger_Left))
        elif move_left < move_right:
            self.tiger_state_index += 1
            self.tiger_state_index = min(self.tiger_state_index, self.real_state_space.index(Tiger_Right))

    def receive_action(self, agent_action_index):
        if self.action_space[agent_action_index] == Open_Left:
            if self.real_state_space[self.tiger_state_index] == Tiger_Left:
                self.tiger_shift()
                return [Eaten_reward, Tiger_Left]
            elif self.real_state_space[self.tiger_state_index] == Tiger_Right:
                self.tiger_shift()
                return [Evade_reward, Tiger_Right]
            else:
                exit('Exception On receive action: OpenLeft!')

        elif self.action_space[agent_action_index] == Open_Right:
            if self.real_state_space[self.tiger_state_index] == Tiger_Left:
                self.tiger_shift()
                return [Evade_reward, Tiger_Right]
            elif self.real_state_space[self.tiger_state_index] == Tiger_Right:
                self.tiger_shift()
                return [Eaten_reward, Tiger_Left]
            else:
                exit('Exception On receive action: OpenRight!')

        elif self.action_space[agent_action_index] == Listen:
            observation = np.random.rand()
            if self.real_state_space[self.tiger_state_index] == Tiger_Left:
                if observation > 0.15:
                    observation = Tiger_Left
                else:
                    observation = Tiger_Right
            elif self.real_state_space[self.tiger_state_index] == Tiger_Right:
                if observation > 0.15:
                    observation = Tiger_Right
                else:
                    observation = Tiger_Left
            return [Listen_reward, observation]
        else:
            exit('Exception On receive action:NONE!')

if __name__ == "__main__":
    TOTAL_REWARD = 0
    EPISODE_REWARD = 0
    maximum_episode = 1
    episode_length = 5
    Agent = agent()
    Env = Environment()
    PSR = PSR()
    testset = PSR.generate_tests()
    U_T = PSR.Computing_U_T()
    U_Q_Name, U_Q = PSR.generate_U_Q()
#    Agent.set_state_dim(state_dim=PSR.gain_core_tests_dim())
#    Agent.Net.origin_Net.load_weights(filepath='origin_net.h5')
#    Agent.Net.target_Net.load_weights(filepath='target_net.h5')
    global state_space
    state_space.append(PSR.return_predictive_state())
    Agent.set_state(pred_state_index=0)
    for j in range(maximum_episode):
        for i in range(episode_length):
            action_idx = Agent.taking_action()
            reward, Agent.tiger_observation = Env.receive_action(agent_action_index=action_idx)
            EPISODE_REWARD += reward
            predictive_state = PSR.update(action_idx=action_idx, observation=Agent.tiger_observation, count=i)
            state_space.append(predictive_state)
            index = len(state_space) - 1
            Agent.replay_memory(s_index=index-1, action_idx=action_idx, r_t=reward, s1_index=index,)
            Agent.set_state(pred_state_index=index)
        print('this' + str(j) +'episode rewards is:', EPISODE_REWARD)
        TOTAL_REWARD += EPISODE_REWARD
    print('after 600 episode, the total reward is:', TOTAL_REWARD)