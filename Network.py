import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Softmax
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Model
from Hyparameter import Hidden_Layer1, Hidden_Layer2, Hidden_Layer3,\
    LearningRate, State_Size, Action_Dim, TAU, Num_Z, z_max, z_min, DecayRate
import keras.backend as K
from keras.losses import kullback_leibler_divergence, mean_absolute_error


class CriticNet(object):
    def __init__(self, state_dim, state_space, action_space, discount_rate):
        self.LearningRate = LearningRate
        self.state_size = State_Size
        self.action_dim = Action_Dim
        self.TAU = TAU
        self.num_atoms = Num_Z
        self.v_max = z_max
        self.v_min = z_min
        self.state_space = state_space
        self.state_dim = state_dim
        self.action_space = action_space
        self.gamma = discount_rate
        self.OPTIMIZER = Adam(lr=self.LearningRate)
        self.delta_z = (np.float(z_max) - np.float(z_min))/Num_Z
        self.z = np.arange(z_min, z_max, self.delta_z, dtype=float)
        self.trainNet = self.TrainNet(inputs=(self.state_dim,))
        self.trainNet.compile(optimizer=Adam(lr=0.01), loss=kullback_leibler_divergence)
        self.origin_Net = self.Create_Network(inputs=(self.state_dim,))
        self.target_Net = self.Create_Network(inputs=(self.state_dim,))
        self.DecayRate = DecayRate
        self.origin_action_optimal = self.settingup_optimal_action(net=self.origin_Net)
        self.target_action_optimal = self.settingup_optimal_action(net=self.target_Net)
        self.projectfunction = self.Projection()
        self.count = 0

    def TrainNet(self, inputs):
        input = Input(inputs)
        X = Dense(Hidden_Layer1, input_dim=self.state_dim, kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  name='common_layer1')(input)
        X = Activation('relu')(X)
        X = Dense(Hidden_Layer2, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                  name='common_layer2')(X)
        X = Activation('relu')(X)
        X = Dense(Hidden_Layer3, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                  name='common_layer3')(X)
        X = Activation('linear')(X)
        output = Dense(self.num_atoms, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                        name='action')(X)
        output = Softmax(axis=-1)(output)
        model = Model(inputs=input, outputs=output)
        return model

    #Updating the Target Network
    def target_train(self):
        net_weights = self.origin_Net.get_weights()
        net_target_weight = self.target_Net.get_weights()
        for i in range(len(net_weights)):
            net_target_weight[i] = (1-TAU) * net_weights[i] + self.TAU * net_target_weight[i]
        self.target_Net.set_weights(weights=net_target_weight)

    #Building up Network Structure
    def Create_Network(self, inputs):
        print("Starting up construction on Network!")
        input = Input(inputs)
        X = Dense(Hidden_Layer1, input_dim=self.state_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  name='common_layer1')(input)
        X = Activation('relu')(X)
        X = Dense(Hidden_Layer2, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                  name='common_layer2')(X)
        X = Activation('relu')(X)
        X = Dense(Hidden_Layer3, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                  name='common_layer3')(X)
        X = Activation('linear')(X)
        output = []
        for i in range(len(self.action_space)):
            X_i = Dense(self.num_atoms, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros',
                        name='action'+str(i))(X)
            X_i = Softmax(axis=-1)(X_i)
            output.append(X_i)
        model = Model(inputs=input, outputs=output)
        return model

    #return the probabilities over z_i with respect to a pair (w,a)
    def reshape_distribution(self, theta_i):
        pro = None
        for i in range(len(theta_i)):
            if pro is None:
                pro = K.expand_dims(x=theta_i[i], axis=1)
            else:
                pro = K.concatenate([pro, K.expand_dims(x=theta_i[i], axis=1)], axis=1)
        return pro

    #return the projected Probability distribution
    def Projection(self):
        reward = K.placeholder(shape=(None,), dtype='float32')
        Pro_Dis = K.placeholder(shape=(None, self.num_atoms), dtype='float64')
        m_prob = K.zeros(shape=(K.shape(reward)[0], self.num_atoms), dtype='float64')
        for j in range(self.num_atoms):
            Tz = K.cast(x=K.minimum(x=self.v_max, y=K.maximum(x=self.v_min, y=reward + self.gamma * self.z[j])), dtype='float64')
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = tf.floor(bj), tf.ceil(bj)
            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, K.shape(reward)[0], 1, dtype='int64'), shape=(-1, 1))
            index_m_l = K.concatenate([temp, m_l_id], axis=-1)
            index_m_u = K.concatenate([temp, m_u_id], axis=-1)
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
            m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
        return K.function([reward, Pro_Dis], [m_prob])


    #return expectation of distribution
    def return_expectation_distribution(self, Dis):
        expectation = self.z * Dis
        exp = K.sum(x=expectation, axis=-1)
        return exp

    # a* = argmax_a Z(a,w)
    def selecting_optimal_action(self, Predictive_State, net):
        input = np.reshape(a=Predictive_State, newshape=(-1, self.state_dim))
        if net == 'origin':
            return self.origin_action_optimal([input])
        elif net == 'target':
            return self.target_action_optimal([input])

    def settingup_optimal_action(self, net):
        input = K.placeholder(shape=(None, self.state_dim), dtype='float32')
        theta_j = net(inputs=input)
        All_Pro_list = self.reshape_distribution(theta_i=theta_j)
        expect_Pro_over_z = self.return_expectation_distribution(Dis=All_Pro_list)
        action_idx = K.argmax(x=expect_Pro_over_z, axis=-1)
        action_id = action_idx
        idx = K.reshape(x=action_idx, shape=(-1, 1))
        temp = K.reshape(x=K.arange(0, K.shape(idx)[0], 1, 'int64'), shape=(-1, 1))
        index_action = K.concatenate([temp, idx])
        Optimal_pro_z = tf.gather_nd(params=All_Pro_list, indices=index_action)
        return K.function([input], [action_id, Optimal_pro_z, expect_Pro_over_z])

    def train_batch(self, Last_Predictive_State, index_action_i, reward_i, Next_Predictive_State):
        index_action, Optimal_action_pro_z, all_Dis = self.selecting_optimal_action(Predictive_State=Next_Predictive_State, net='target')
        Projected_update_current_state_pro_z = self.projectfunction([reward_i, Optimal_action_pro_z])
        Projected_update_current_state_pro_z = np.array(Projected_update_current_state_pro_z[0])
        print('The sum of probability of Projected distribution' + str(np.sum(a=Projected_update_current_state_pro_z, axis=-1)))
        print('the Optimal action on Next State'+str(index_action))
        print('the reward is:'+str(reward_i))
        print('the action:'+str(index_action_i))
        self.Custom_loss(Predictive_State=Last_Predictive_State, index_action_i=index_action_i, upd_distribution=Projected_update_current_state_pro_z)

    def _train(self, samples):
        samples = np.array(samples)
        samples = sorted(samples, key=lambda x: x[1])
        samples = np.array(samples)
        a0_idx = samples[samples[:, 1] == 0]
        a1_idx = samples[samples[:, 1] == 1]
        a2_idx = samples[samples[:, 1] == 2]
        sampleArray = [a0_idx, a1_idx, a2_idx]

        for i in range(len(sampleArray)):
            Last_Predictive_State = sampleArray[i][:, 0]
            Last_Predictive_State = np.array(list(Last_Predictive_State)) * 100.0
            index_action = sampleArray[i][:, 1]
            index_action = index_action.astype('int64')
            rewards = sampleArray[i][:, 2]
            Next_Predictive_State = sampleArray[i][:, 3]
            Next_Predictive_State = np.array(list(Next_Predictive_State)) * 100.0
            if len(index_action) != 0:
                self.train_batch(Last_Predictive_State=Last_Predictive_State, index_action_i=index_action, reward_i=rewards, Next_Predictive_State=Next_Predictive_State)

    def CopyWeight(self, a_id):
        for i in range(len(self.trainNet.layers)-2):
            self.trainNet.layers[i].set_weights(self.origin_Net.layers[i].get_weights())
        self.trainNet.layers[7].set_weights(self.origin_Net.layers[a_id+7].get_weights())
        self.trainNet.layers[8].set_weights(self.origin_Net.layers[a_id+10].get_weights())

    def SetWeight(self, a_id):
        for i in range(len(self.trainNet.layers)-2):
            self.origin_Net.layers[i].set_weights(self.trainNet.layers[i].get_weights())
        self.origin_Net.layers[a_id+7].set_weights(self.trainNet.layers[7].get_weights())
        self.origin_Net.layers[a_id+10].set_weights(self.trainNet.layers[8].get_weights())

    # return loss in this batch
    def Custom_loss(self, Predictive_State, index_action_i, upd_distribution):
        input = np.reshape(a=Predictive_State, newshape=(-1, self.state_dim))
        self.CopyWeight(a_id=index_action_i[0])
        loss = self.trainNet.train_on_batch(x=input, y=upd_distribution)
        print('the loss on this update is', loss)
        self.SetWeight(a_id=index_action_i[0])

