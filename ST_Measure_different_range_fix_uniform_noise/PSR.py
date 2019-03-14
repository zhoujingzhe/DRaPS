import numpy as np
import pandas as pd
def length(test):
    return test.count('a')

#A |n|x|s| b |s| x 1
def solve(A,b):
    A = np.array(A)
    b = np.reshape(a=b, newshape=(-1, 1))
    res = np.linalg.solve(A, b)
    res = np.reshape(a=res, newshape=(-1,))
    return res

class PSR(object):
    def __init__(self, T, O, b_h, Observations, Actions, R_Matrix):
        self.U_t_Name = []
        self.U_t = []
        self.U_q = []
        self.U_Q = None
        self.U_Q_Name = []
        self.Predictive_State = None
        self.T = T
        self.O = O
        self.R_Matrix = R_Matrix
        self.m = []
        self.m_name = []
        self.M = []
        self.M_name = []
        self.Observations = Observations
        self.Actions = Actions
        self.MaxNumCoreTest = len(b_h[0])
        self.R_list = list(self.R_Matrix.keys())
        self.b_h = b_h

    # generate a test representation
    def generate_test(self, base_sequence, a_id, o_id, r_id):
        aor = 'a'+str(a_id)+'o'+str(o_id) +'r'+str(r_id)
        base_sequence = base_sequence + aor
        return base_sequence

    # producing a sequences of tests representations
    def generate_tests(self, num=100):
        if len(self.U_t_Name) == 0:
            for i in range(len(self.Actions)):
                for j in range(len(self.Observations)):
                    for k in range(len(self.R_list)):
                        self.U_t_Name.append(self.generate_test(base_sequence='', a_id=i, o_id=j, r_id=k))
        count = len(self.U_t_Name)
        k=0
        while count < num:
            base_seq = self.U_t_Name[k]
            k=k+1
            for i in range(len(self.Actions)):
                for j in range(len(self.Observations)):
                    for k in range(len(self.R_list)):
                        self.U_t_Name.append(self.generate_test(base_sequence=base_seq, a_id=i, o_id=j, r_id=k))
                        count = len(self.U_t_Name)
                        if count >= num:
                            return self.U_t_Name
        return self.U_t_Name

    # Computing the probabilities of tests
    def Computing_U_T(self):
        for i in range(len(self.U_t_Name)):
            U_t_Name = self.U_t_Name[i]
            U_t = np.identity(len(self.Observations))
            iters = np.arange(0, len(U_t_Name), 6)
            for j in iters:
                if U_t_Name[j] == 'a':
                    num_a = np.int(x=U_t_Name[j+1])
                    num_o = np.int(x=U_t_Name[j+3])
                    num_r = np.int(x=U_t_Name[j+5])
                    T = self.T[num_a]
                    O = self.O[num_a][num_o]
                    R = self.R_Matrix[self.R_list[num_r]][num_a, :, :, num_o]
                    U_t = np.matmul(a=U_t, b=np.multiply(np.matmul(T, O), R))
                else:
                    print('exception on Computing tests')
            e = np.ones(shape=(np.shape(a=U_t)[1], 1))
            U_t = np.matmul(a=U_t, b=e)
            self.U_t.append(U_t)
        return self.U_t

    # produce core tests
    def generate_U_Q(self):
        rank = 0
        for i in range(len(self.U_t)):
            self.U_q.append(np.reshape(a=self.U_t[i], newshape=(-1)))
            if np.linalg.matrix_rank(self.U_q) > rank:
                self.U_Q_Name.append(self.U_t_Name[i])
                rank = np.linalg.matrix_rank(self.U_q)
            else:
                self.U_q.pop(-1)
            if rank >= self.MaxNumCoreTest:
                self.U_Q = np.transpose(self.U_q)
                return [self.U_Q_Name, self.U_Q]
        self.U_Q = np.transpose(self.U_q)
        return [self.U_Q_Name, self.U_Q]

    # obtain m_ao for all possible a and o
    def gain_m(self):
        for i in range(len(self.U_t_Name)):
            test = self.U_t_Name[i]
            if length(test) == 1:
                m_ao = solve(self.U_Q, self.U_t[i])
                m_ao = np.reshape(a=m_ao, newshape=(-1, 1))
                self.m.append(m_ao)
                self.m_name.append(test)
        return [self.m_name, self.m]

    # calculating U_t for a specific test t
    def gain_U_t(self, test_name):
        iters = np.arange(0, len(test_name), 6)
        test_pro = np.identity(len(self.Observations))
        for j in iters:
            if test_name[j] == 'a':
                num_a = np.int(x=test_name[j + 1])
                num_o = np.int(x=test_name[j + 3])
                num_r = np.int(x=test_name[j + 5])
                T = self.T[num_a]
                O = self.O[num_a][num_o]
                R = self.R_Matrix[self.R_list[num_r]][num_a, :, :, num_o]
                test_pro = np.matmul(a=test_pro, b=np.multiply(np.matmul(T, O), R))
            else:
                print('exception on Computing tests')
        e = np.ones(shape=(np.shape(a=test_pro)[1], 1))
        test_pro = np.matmul(a=test_pro, b=e)
        return test_pro

    # obtain M_ao for a specific ao
    def gain_M_ao(self):
        for i in range(len(self.U_t_Name)):
            test = self.U_t_Name[i]
            if length(test) == 1:
                M_ao = []
                M_ao_name = []
                for i in range(len(self.U_Q_Name)):
                    M_ao.append(solve(self.U_Q, self.gain_U_t(test_name=test+self.U_Q_Name[i])))
                    M_ao_name.append(self.U_Q_Name[i])
                M_ao = np.array(M_ao).T
                self.M.append(M_ao)
                self.M_name.append(test)

    def update_batch(self, a_id, o_id, r_id, Predictive_State):
        Predictive_State = np.array(list(Predictive_State))
        a_id = np.array(a_id, dtype=np.str)
        o_id = np.array(o_id, dtype=np.str)
        r_id = np.array(r_id, dtype=np.str)
        a = np.broadcast_to(array='a', shape=np.shape(a_id))
        o = np.broadcast_to(array='o', shape=np.shape(o_id))
        r = np.broadcast_to(array='r', shape=np.shape(r_id))
        a_id = np.core.defchararray.add(a, a_id)
        o_id = np.core.defchararray.add(o, o_id)
        r_id = np.core.defchararray.add(r, r_id)
        test = np.core.defchararray.add(a_id, np.core.defchararray.add(o_id, r_id))
        index = np.searchsorted(self.m_name, test)
        m_ao = np.array(self.m)[index]
        index = np.searchsorted(self.M_name, test)
        M_ao = np.array(self.M)[index]
        TPredictive_State = np.transpose(a=Predictive_State, axes=(0, 2, 1))
        denominator = np.matmul(a=TPredictive_State, b=m_ao)
        numerator = np.matmul(a=TPredictive_State, b=M_ao)
        Tnew_Predictive_State = numerator / (denominator + np.finfo(float).eps)
        new_Predictive_State = np.transpose(a=Tnew_Predictive_State, axes=[0, 2, 1])
        return [new_Predictive_State, denominator]

    def update(self, action_idx, observation_id, r_id):
        test_ao = 'a' + str(action_idx) + 'o' + str(observation_id) + 'r' + str(r_id)
        if test_ao not in self.m_name:
            print('exception on m_ao')
        index = self.m_name.index(test_ao)
        m_ao = self.m[index]
        m_ao = np.reshape(a=m_ao, newshape=(-1,))
        if test_ao not in self.M_name:
            print('exception on M_ao')
        M_ao = self.M[self.M_name.index(test_ao)]
        if self.Predictive_State is None:
            self.Predictive_State = self.return_predictive_state()
        denominator = np.matmul(a=self.Predictive_State.T, b=m_ao) + np.finfo(float).eps
        numerator = np.matmul(a=self.Predictive_State.T, b=M_ao)
        new_Predictive_State = numerator/denominator
        self.Predictive_State = new_Predictive_State.T
        return self.Predictive_State

    # return predictive state P(Q|h)
    def return_predictive_state(self):
        U_Q = self.U_Q
        b_h = self.b_h
        return np.matmul(a=b_h, b=U_Q).T

    def gain_core_tests_dim(self):
        return np.prod(np.shape(self.Predictive_State), axis=-1)

    # print all details
    def print_all(self, File_name):
        P_Q = pd.DataFrame(data=self.Predictive_State)
        U_Q = pd.DataFrame(data=self.U_Q, columns=self.U_Q_Name)
        M_ao = pd.DataFrame.from_records(data=self.M, index=self.M_name)
        m_ao = pd.DataFrame.from_records(data=self.m, index=self.m_name)
        P_Q.to_csv(path_or_buf=File_name+'.csv', mode='a')
        U_Q.to_csv(path_or_buf=File_name+'.csv', mode='a')
        M_ao.to_csv(path_or_buf=File_name+'.csv', mode='a')
        m_ao.to_csv(path_or_buf=File_name+'.csv', mode='a')

from Environment import POMDPEnvironment
if __name__ == "__main__":
    #####initialization######
    EnvObject = POMDPEnvironment(filename='tiger.95.POMDP')
    T = EnvObject._obtain_transition()
    O = EnvObject._obtain_observation()
    R_Matrix = EnvObject._obtain_reward_matrix()
    b_h = EnvObject._obtain_b_h()
    b_h = np.reshape(a=b_h, newshape=(1, -1))
    discount_rate = EnvObject.discount
    Observations = EnvObject.observations
    States = EnvObject.states
    Actions = EnvObject.actions
    ###########################################
    PSR = PSR(T=T, O=O, b_h=b_h, Observations=Observations, Actions=Actions, R_Matrix=R_Matrix)
    testset = PSR.generate_tests()
    U_T = PSR.Computing_U_T()
    U_Q_Name, U_Q = PSR.generate_U_Q()
    PSR.gain_m()
    for i in range(len(Actions)):
        for j in range(len(Observations)):
            for k in range(len(PSR.R_list)):
                PSR.gain_M_ao(ao='a'+str(i)+'o'+str(j)+'r'+str(k))
    PSR.print_all('Detail')