import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from Hyparameter import Hidden_Layer1, Hidden_Layer2, Hidden_Layer3,\
    LearningRate, State_Size, Action_Dim, TAU, Num_Z, z_max, z_min, batch_size, state_space, action_space, discount_rate
import keras.backend as K
import math
from keras.losses import categorical_crossentropy

class CriticNet(object):
    def __init__(self, state_dim):
        self.LearningRate = LearningRate
        self.state_size = State_Size
        self.action_dim = Action_Dim
        self.TAU = TAU
        self.num_atoms = Num_Z
        self.v_max = z_max
        self.v_min = z_min
        self.batch_size = batch_size
        self.state_space = state_space
        self.state_dim = state_dim
        self.action_space = action_space
        self.gamma = discount_rate
        self.delta_z = (np.float(z_max) - np.float(z_min))/Num_Z
        self.z = np.arange(z_min, z_max, self.delta_z, dtype=float)
        self.origin_Net = self.Create_Network(inputs=(self.state_dim,))
        self.target_Net = self.Create_Network(inputs=(self.state_dim,))
        self.optimizer = Adam(lr=self.LearningRate)

    #Updating the Target Network
    def target_train(self):
        net_weights = self.origin_Net.get_weights()
        net_target_weight = self.target_Net.get_weights()
        for i in range(len(net_weights)):
            net_target_weight[i] = self.TAU * net_weights[i] + (1-TAU) * net_target_weight[i]
        self.target_Net.set_weights(weights=net_target_weight)

    #Building up Network Structure
    def Create_Network(self, inputs):
        print("Starting up construction on Network!")
        input = Input(inputs)
        X = Dense(Hidden_Layer1, input_dim=self.state_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input)
        X = Activation('relu')(X)
        X = Dense(Hidden_Layer2, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros')(X)
        X = Activation('linear')(X)
        output = []
        for i in range(len(action_space)):
            X_i = Dense(Hidden_Layer3, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros')(X)
            X_i = LeakyReLU(alpha=0.3)(X_i)
            output.append(X_i)
        model = Model(inputs=input, outputs=output)
        return model

    #return the probabilities over z_i with a given pair (x,a)
    def _probability(self, theta_i):
        pro = K.softmax(x=theta_i, axis=-1)
        newpro = []
        e = K.eval(K.shape(x=theta_i)[1])
        for i in range(e):
            temp = pro[:, i, :]
            newpro.append(temp)
        newpro = tf.convert_to_tensor(newpro)
        return newpro

    #return the projected Probability distribution over all z_i
    def Projection(self, reward, Pro_Dis, batch_size):
        m_prob = K.zeros(shape=K.shape(Pro_Dis))
        for j in range(self.num_atoms):
            Tz = K.minimum(x=K.constant(value=self.v_max, dtype='float64'), y=K.maximum(x=K.constant(value=self.v_min, dtype='float64'), y=reward + self.gamma * self.z[j]))
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = tf.floor(bj), tf.ceil(bj)
            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, batch_size, 1, dtype='int64'), shape=(-1, 1))
            m_l_id = K.concatenate([temp, m_l_id], axis=-1)
            m_u_id = K.concatenate([temp, m_u_id], axis=-1)
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            tmp1 = K.cast(x=tf.gather_nd(params=m_prob, indices=m_l_id), dtype='float64') + tmp1
            tmp2 = K.cast(x=tf.gather_nd(params=m_prob, indices=m_u_id), dtype='float64') + tmp2
            m_prob = K.cast(x=m_prob, dtype='float64') + tf.scatter_nd(indices=m_l_id, updates=tmp1, shape=K.cast(x=K.shape(m_prob), dtype='int64'))
            m_prob = K.cast(x=m_prob, dtype='float64') + tf.scatter_nd(indices=m_u_id, updates=tmp2, shape=K.cast(x=K.shape(m_prob), dtype='int64'))
        return m_prob

    #return expectation of distribution
    def return_expectation_distribution(self, Dis):
        expectation = self.z * Dis
        exp = K.sum(x=expectation, axis=-1)
        return exp

    def selecting_optimal_action(self, index_state, net, batch_size):
        input = K.cast(K.reshape(x=np.array(state_space)[index_state], shape=(-1, self.state_dim)), dtype='float32')
        theta_j = net(inputs=input)
        All_Pro_list = self._probability(theta_i=theta_j)
        expect_Pro_over_z = self.return_expectation_distribution(Dis=All_Pro_list)
        action_idx = K.argmax(x=expect_Pro_over_z, axis=-1)
        action_id = action_idx
        action_idx = K.reshape(x=action_idx, shape=(-1,1))
        action_idx = K.concatenate([K.reshape(x=K.arange(0, batch_size, 1, dtype='int64'), shape=(batch_size, 1)), action_idx], axis=1)
        Optimal_pro_z = tf.gather_nd(params=All_Pro_list, indices=action_idx)
        return [K.eval(action_id), K.eval(Optimal_pro_z)]

    def train_batch(self, index_state_i, index_action_i, reward_i, index_state_i_plus_one, batch_size):
        index_action, Optimal_action_pro_z = self.selecting_optimal_action(index_state=index_state_i_plus_one, net=self.target_Net, batch_size=batch_size)
        Projected_update_current_state_pro_z = self.Projection(reward=reward_i, Pro_Dis=Optimal_action_pro_z, batch_size=batch_size)
        loss = self.Custom_loss(index_state_i=index_state_i, index_action_i=index_action_i, upd_distribution=Projected_update_current_state_pro_z, batch_size=batch_size)
        loss = K.mean(x=loss, axis=-1)
        self.optimizer.get_updates(loss=loss, params=self.origin_Net.trainable_weights)

    def _train(self, samples, batch_size):
        samples = np.array(samples)
        np.random.shuffle(samples)
        index_state_current = samples[:, 0]
        index_action = samples[:, 1]
        rewards = samples[:, 2]
        index_state_next = samples[:, 3]
        i = 0
        while i < len(samples):
            self.train_batch(index_state_i=index_state_current[i:i+batch_size], index_action_i=index_action[i:i+batch_size], reward_i=rewards[i:i+batch_size], index_state_i_plus_one=index_state_next[i:i+batch_size], batch_size=batch_size)
            i = i + batch_size

    def Custom_loss(self, index_state_i, index_action_i, upd_distribution, batch_size):
        input = K.cast(K.reshape(x=np.array(state_space)[index_state_i], shape=(-1, self.state_dim)), dtype='float32')
        ori_distributions = self.origin_Net(inputs=input)
        ori_distributions = self._probability(theta_i=ori_distributions)
        temp = K.reshape(x=K.arange(0,batch_size,1,dtype='int64'), shape=(-1,1))
        index_action_i = K.reshape(x=index_action_i, shape=(-1,1))
        index_action_i = K.concatenate([temp, index_action_i], axis=-1)
        ori_distribution = K.cast(x=tf.gather_nd(params=ori_distributions, indices=index_action_i), dtype = 'float64')
        loss = categorical_crossentropy(y_true=upd_distribution, y_pred=ori_distribution)
        return loss
