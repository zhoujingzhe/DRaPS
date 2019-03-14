import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Softmax
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Model
from Hyparameter import Hidden_Layer1, Hidden_Layer2, Hidden_Layer3,\
    LearningRate, TAU, Num_Z, z_max, z_min, alpha
import keras.backend as K
from keras.losses import kullback_leibler_divergence
def KL_loss():
    upd_distribution = K.placeholder(shape=(None, 232), dtype='float64')
    ori_distritbuion = K.placeholder(shape=(None, 232), dtype='float64')
    upd_distribution = K.clip(x=upd_distribution, min_value=K.epsilon(), max_value=1)
    ori_distritbuion = K.clip(x=ori_distritbuion, min_value=K.epsilon(), max_value=1)
    kl_batch_loss = K.sum(x=upd_distribution * K.log(x=upd_distribution/ori_distritbuion), axis=-1)
    kl_loss = K.mean(x=kl_batch_loss, axis=-1)
    train = K.function(inputs=[ori_distritbuion, upd_distribution], outputs=[kl_loss, kl_batch_loss])
    return train
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def f(m_u):
    K.set_value(x=m_u, value=1)
    return m_u

class CriticNet(object):
    def __init__(self, state_dim, state_space, action_space, discount_rate, length_or):
        self.kl_function = KL_loss()
        self.length_or = length_or
        self.LearningRate = LearningRate
        self.action_dim = len(action_space)
        self.TAU = TAU
        self.num_atoms = Num_Z
        self.v_max = np.float64(z_max)
        self.v_min = np.float64(z_min)
        self.state_space = state_space
        self.state_dim = state_dim
        self.action_space = action_space
        self.gamma = np.float64(discount_rate)
        self.delta_z = (np.float64(z_max) - np.float64(z_min))/Num_Z
        self.z = np.arange(z_min, z_max, self.delta_z, dtype=float)
        self.trainNet = self.TrainNet(inputs=(self.state_dim,))
        self.trainNet.compile(optimizer=Adam(lr=0.001), loss=kullback_leibler_divergence)
        self.origin_Net = self.Create_Network(inputs=(self.state_dim,))
        self.target_Net = self.Create_Network(inputs=(self.state_dim,))
        self.origin_action_optimal = self.settingup_optimal_action(net=self.origin_Net)
        self.target_action_optimal = self.settingup_optimal_action(net=self.target_Net)
        self.projectfunction = self.Projection_add_noise()
        self.count = 0
        self.alpha = alpha

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


    #pure projection without noise
    def Projection(self):
        reward = K.placeholder(shape=(None,), dtype='float64')
        Pro_Dis = K.placeholder(shape=(None, self.num_atoms), dtype='float64')
        m_prob = K.zeros(shape=(K.shape(reward)[0], self.num_atoms), dtype='float64')
        alpha = K.placeholder(shape=(1, ), dtype='float64')
        for j in range(self.num_atoms):
            Tz = K.cast(x=K.minimum(x=self.v_max, y=K.maximum(x=self.v_min, y=reward + self.gamma * self.z[j])), dtype='float64')
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = tf.floor(bj), tf.ceil(bj)
            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, K.shape(reward)[0], 1, dtype='int64'), shape=(-1, 1))
            index_m_l = K.concatenate([temp, m_l_id], axis=-1)
            index_m_u = K.concatenate([temp, m_u_id], axis=-1)
            cond = K.equal(x=m_u, y=0)
            m_u = K.cast(x=cond, dtype='float64') + m_u
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
            m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
        return K.function([reward, Pro_Dis, alpha], [m_prob])

    #return the projected Probability distribution
    def Projection_add_noise(self):
        reward = K.placeholder(shape=(None,), dtype='float64')
        Pro_Dis = K.placeholder(shape=(None, self.num_atoms), dtype='float64')
        m_prob = K.zeros(shape=(K.shape(reward)[0], self.num_atoms), dtype='float64')
        alpha = K.placeholder(shape=(1, ), dtype='float64')
        for j in range(self.num_atoms):
            Tz = K.cast(x=K.minimum(x=self.v_max, y=K.maximum(x=self.v_min, y=reward + self.gamma * self.z[j])), dtype='float64')
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = tf.floor(bj), tf.ceil(bj)
            m_l_id = K.reshape(x=K.cast(x=m_l, dtype='int64'), shape=(-1, 1))
            m_u_id = K.reshape(x=K.cast(x=m_u, dtype='int64'), shape=(-1, 1))
            temp = K.reshape(x=K.arange(0, K.shape(reward)[0], 1, dtype='int64'), shape=(-1, 1))
            index_m_l = K.concatenate([temp, m_l_id], axis=-1)
            index_m_u = K.concatenate([temp, m_u_id], axis=-1)
            cond = K.equal(x=m_u, y=0)
            m_u = K.cast(x=cond, dtype='float64') + m_u
            tmp1 = Pro_Dis[:, j] * (m_u - bj)
            tmp2 = Pro_Dis[:, j] * (bj - m_l)
            m_prob = m_prob + tf.scatter_nd(indices=index_m_l, updates=tmp1, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
            m_prob = m_prob + tf.scatter_nd(indices=index_m_u, updates=tmp2, shape=K.cast(x=(K.shape(reward)[0], self.num_atoms), dtype='int64'))
        uniform_noise = K.ones(shape=K.shape(m_prob), dtype='float64')
        num_atom = K.cast(x=K.shape(m_prob)[1], dtype='float64')
        uniform_noise = uniform_noise / num_atom
        m_prob = alpha * m_prob + (1 - alpha) * uniform_noise
        return K.function([reward, Pro_Dis, alpha], [m_prob])


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

    def train_batch(self, Last_Predictive_State, index_action_i, reward_i, Next_Predictive_State, Probability):
        index_action, Optimal_action_pro_z, all_Dis = self.selecting_optimal_action(Predictive_State=Next_Predictive_State, net='target')
        if self.count % 5000 == 0:
            self.alpha = 1 - (1 - self.alpha)*0.8
        Projected_update_current_state_pro_z = self.projectfunction([reward_i, Optimal_action_pro_z, self.alpha])
        Projected_update_current_state_pro_z = np.array(Projected_update_current_state_pro_z[0])
        Probability = np.reshape(a=Probability, newshape=(-1, 1))
        temp = Projected_update_current_state_pro_z * Probability
        Next_State_Distribution = []
        _Last_Predictive_State = []
        _index_action = []
        i = 0
        while(i<len(temp)):
            a = temp[i:i+self.length_or:1]
            _Last_Predictive_State.append(Last_Predictive_State[i])
            _index_action.append([index_action_i[i]])
            Next_State_Distribution.append(np.sum(a=a, axis=0))
            i = i + self.length_or

        ########################################################################
        #test block
        Next_State_Distribution = np.float32(Next_State_Distribution)
        s = np.sum(a=Next_State_Distribution, axis=-1)
        s = np.reshape(a=s, newshape=(-1, 1))
        Next_State_Distribution = Next_State_Distribution / s
        #########################################################################
        self.Custom_loss(Predictive_State=_Last_Predictive_State, index_action_i=_index_action, upd_distribution=Next_State_Distribution)

    def _train(self, samples):
        # record the times of updating
        self.count = self.count + 1
        samples = np.array(samples)
        samples = sorted(samples, key=lambda x: x[1])
        samples = np.array(samples)
        sampleArray = []
        for i in range(self.action_dim):
            sampleArray.append(samples[samples[:, 1] == i])

        for i in range(len(sampleArray)):
            Last_Predictive_State = sampleArray[i][:, 0]
            Last_Predictive_State = np.array(list(Last_Predictive_State))
            index_action = sampleArray[i][:, 1]
            index_action = index_action.astype('int64')
            R = sampleArray[i][:, 4]
            Probability = sampleArray[i][:, 5]
            Next_Predictive_State = sampleArray[i][:, 6:]

            if len(index_action) != 0:
                self.train_batch(Last_Predictive_State=Last_Predictive_State, index_action_i=index_action, reward_i=R, Probability=Probability, Next_Predictive_State=Next_Predictive_State)

    def CopyWeight(self, a_id):
        for i in range(len(self.trainNet.layers)-2):
            self.trainNet.layers[i].set_weights(self.origin_Net.layers[i].get_weights())
        self.trainNet.layers[7].set_weights(self.origin_Net.layers[a_id+7].get_weights())
        self.trainNet.layers[8].set_weights(self.origin_Net.layers[a_id+7+len(self.action_space)].get_weights())

    def SetWeight(self, a_id):
        for i in range(len(self.trainNet.layers)-2):
            self.origin_Net.layers[i].set_weights(self.trainNet.layers[i].get_weights())
        self.origin_Net.layers[a_id+7].set_weights(self.trainNet.layers[7].get_weights())
        self.origin_Net.layers[a_id+7+len(self.action_space)].set_weights(self.trainNet.layers[8].get_weights())

    # return loss in this batch
    def Custom_loss(self, Predictive_State, index_action_i, upd_distribution):
        input = np.reshape(a=Predictive_State, newshape=(-1, self.state_dim))
        upd_distribution = np.reshape(a=upd_distribution, newshape=(-1, self.num_atoms))
        self.CopyWeight(a_id=index_action_i[0][0])
        ori_distribution = self.trainNet.predict(x=input, batch_size=len(input))
        p1 = np.sum(a=ori_distribution, axis=-1)
        loss = self.trainNet.train_on_batch(x=input, y=upd_distribution)
        if loss < 0:
            print('error')
        print('the loss on this update is', loss)
        self.SetWeight(a_id=index_action_i[0][0])

