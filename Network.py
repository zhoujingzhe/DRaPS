import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from Hyparameter import Hidden_Layer1, Hidden_Layer2, Hidden_Layer3,\
    LearningRate, State_Size, Action_Dim, TAU, Num_Z, z_max, z_min, batch_size, predictive_state_space
import keras.backend as K
from keras.losses import categorical_crossentropy


class CriticNet(object):
    def __init__(self, state_dim, state_space, action_space, discount_rate):
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
        self.optimizer = self.Update_Weight()
        self.origin_action_optimal = self.optimal_action_on_origin()
        self.target_action_optimal = self.optimal_action_on_target()
        self.projectfunction = self.Projection()



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
        for i in range(len(self.action_space)):
            X_i = Dense(Hidden_Layer3, input_dim=K.shape(X), kernel_initializer='random_uniform', bias_initializer='zeros')(X)
            X_i = LeakyReLU(alpha=0.3)(X_i)
            output.append(X_i)
        model = Model(inputs=input, outputs=output)
        return model

    #return the probabilities over z_i with respect to a pair (w,a)
    def _probability(self, theta_i):
        pro = K.softmax(x=theta_i, axis=-1)
        newpro = []
        e = K.int_shape(pro)[1]
        for i in range(e):
            temp = pro[:, i, :]
            newpro.append(temp)
        newpro = tf.convert_to_tensor(newpro)
        return newpro

    #return the projected Probability distribution
    def Projection(self):
        m_prob = K.zeros(shape=(self.batch_size, self.num_atoms), dtype='float64')
        reward = K.placeholder(shape=(self.batch_size,), dtype='float32')
        Pro_Dis = K.placeholder(shape=(self.batch_size, self.num_atoms), dtype='float64')
        for j in range(self.num_atoms):
            Tz = K.cast(x=K.minimum(x=self.v_max, y=K.maximum(x=self.v_min, y=reward + self.gamma * self.z[j])),dtype='float64')
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = tf.floor(bj), tf.ceil(bj)
            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, self.batch_size, 1, dtype='int64'), shape=(-1, 1))
            index_m_l = K.concatenate([temp, m_l_id], axis=-1)
            index_m_u = K.concatenate([temp, m_u_id], axis=-1)
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1, shape=K.cast(x=(self.batch_size, self.num_atoms), dtype='int64'))
            m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2, shape=K.cast(x=(self.batch_size, self.num_atoms), dtype='int64'))
        return K.function([reward, Pro_Dis], [m_prob])


    #return expectation of distribution
    def return_expectation_distribution(self, Dis):
        expectation = self.z * Dis
        exp = K.sum(x=expectation, axis=-1)
        return exp

    # a* = argmax_a Z(a,w)
    def selecting_optimal_action(self, index_state, net):
        input = K.cast(K.reshape(x=np.array(predictive_state_space)[index_state], shape=(-1, self.state_dim)), dtype='float32')
        temp = K.reshape(x=K.arange(0, np.shape(index_state)[0], 1, dtype='int64'), shape=(-1, 1))
        if net == 'origin':
            return self.origin_action_optimal([input, temp])
        elif net == 'target':
            return self.target_action_optimal([input, temp])

    def optimal_action_on_target(self):
        input = K.placeholder(shape=(None, self.state_dim), dtype='float32')
        theta_j = self.target_Net(inputs=input)
        All_Pro_list = self._probability(theta_i=theta_j)
        expect_Pro_over_z = self.return_expectation_distribution(Dis=All_Pro_list)
        action_idx = K.argmax(x=expect_Pro_over_z, axis=-1)
        action_id = action_idx
        action_idx = K.reshape(x=action_idx, shape=(-1, 1))
        temp = K.placeholder(shape=(None, 1), dtype='int64')
        index_action = K.concatenate([temp, action_idx])
        Optimal_pro_z = tf.gather_nd(params=All_Pro_list, indices=index_action)
        return K.function([input, temp], [action_id, Optimal_pro_z])


    def optimal_action_on_origin(self):
        input = K.placeholder(shape=(None, self.state_dim), dtype='float32')
        theta_j = self.origin_Net(inputs=input)
        All_Pro_list = self._probability(theta_i=theta_j)
        expect_Pro_over_z = self.return_expectation_distribution(Dis=All_Pro_list)
        action_idx = K.argmax(x=expect_Pro_over_z, axis=-1)
        action_id = action_idx
        action_idx = K.reshape(x=action_idx, shape=(-1, 1))
        temp = K.placeholder(shape=(None, 1), dtype='int64')
        index_action = K.concatenate([temp, action_idx])
        Optimal_pro_z = tf.gather_nd(params=All_Pro_list, indices=index_action)
        return K.function([input, temp], [action_id, Optimal_pro_z])

    def train_batch(self, index_state_i, index_action_i, reward_i, index_state_i_plus_one):
        index_action, Optimal_action_pro_z = self.selecting_optimal_action(index_state=index_state_i_plus_one, net='target')
        import time
        tick1 = time.time()
        Projected_update_current_state_pro_z = self.projectfunction([reward_i, Optimal_action_pro_z])
        tick2 = time.time()
        self.Custom_loss(index_state_i=index_state_i, index_action_i=index_action_i, upd_distribution=Projected_update_current_state_pro_z)
        tick3 = time.time()
        print(tick2-tick1)
        print(tick3-tick2)


    def _train(self, samples, batch_size):
        samples = np.array(samples)
        np.random.shuffle(samples)
        index_state_current = samples[:, 0]
        index_state_current = index_state_current.astype('int64')
        index_action = samples[:, 1]
        index_action = index_action.astype('int64')
        rewards = samples[:, 2]
        index_state_next = samples[:, 3]
        index_state_next = index_state_next.astype('int64')
        i = 0
        while i < len(samples):
            self.train_batch(index_state_i=index_state_current[i:i+batch_size], index_action_i=index_action[i:i+batch_size], reward_i=rewards[i:i+batch_size], index_state_i_plus_one=index_state_next[i:i+batch_size])
            i = i + batch_size

    # return loss in this batch
    def Custom_loss(self, index_state_i, index_action_i, upd_distribution):
        input = K.cast(K.reshape(x=np.array(predictive_state_space)[index_state_i], shape=(-1, self.state_dim)), dtype='float32')
        self.optimizer([input, index_action_i, upd_distribution])

    def Update_Weight(self):
        input = K.placeholder(shape=(self.batch_size, self.state_dim), dtype='float32')
        index_action_i = K.placeholder(shape=(self.batch_size,), dtype='int64')
        upd_distribution = K.placeholder(shape=(self.batch_size, Num_Z), dtype='float64')
        ori_distributions = self.origin_Net(inputs=input)
        ori_distributions = self._probability(theta_i=ori_distributions)
        temp = K.arange(0, self.batch_size, 1, dtype='int64')
        temp = K.reshape(x=temp, shape=(self.batch_size, 1))
        index = K.reshape(x=index_action_i, shape=(self.batch_size, 1))
        index = K.concatenate([temp, index])
        ori_distribution = K.cast(x=tf.gather_nd(params=ori_distributions, indices=index), dtype='float64')
        loss = categorical_crossentropy(y_true=upd_distribution, y_pred=ori_distribution)
        loss = K.mean(x=loss, axis=-1)
        Optimizer = Adam(lr=self.LearningRate)
        updates = Optimizer.get_updates(loss=loss, params=self.origin_Net.trainable_weights)
        train = K.function([input, index_action_i, upd_distribution], [], updates=updates)
        return train

