import numpy as np
from Hyparameter import Observation_space, action_space, Open_Left, Open_Right, Listen, Tiger_Left, Tiger_Right, MaxCTest, b_h
import pandas as pd
def length(test):
    return test.count('a')

#A |n|x|s| b |s| x 1
def solve(A,b):
    A = np.array(A) + np.identity(n=2) * np.finfo(float).eps * 1000
    b = np.reshape(a=b, newshape=(-1, 1))
    res = np.linalg.solve(A, b)
    res = np.reshape(a=res, newshape=(-1,))
    return res

class PSR(object):
    def __init__(self):
        self.U_t_Name = []
        self.U_t = []
        self.U_q = []
        self.U_Q = None
        self.U_Q_Name = []
        self.Predictive_State = None
        self.T = []
        T_a1 = [[0.5, 0.5], [0.5, 0.5]] # taking actions open-left and open right
        T_a0 = [[1, 0], [0, 1]] # taking action listen
        self.T.append(T_a1)
        self.T.append(T_a0)
        self.T.append(T_a1)
        self.O = []
        O_1 = [[0.85, 0], [0, 0.15]] # taking action listen, tiger left
        O_2 = [[0.15, 0], [0, 0.85]] # taking action listen, tiger right
        O_3 = [[0.5, 0], [0, 0.5]] # taking action open-left and open right, and seeing tiger left and right
        self.O.append([O_3, O_3])
        self.O.append([O_1, O_2])
        self.O.append([O_3, O_3])
        self.m = []
        self.m_name = []
        self.M = []
        self.M_name = []
        self.Observations = Observation_space
        self.Actions = action_space
        self.MaxNumCoreTest = MaxCTest
        self.b_h = b_h

    # generate a test representation
    def generate_test(self, base_sequence, action, observation):
        act = ''
        if action == Open_Left:
            act = 'a0'
        elif action == Listen:
            act = 'a1'
        elif action == Open_Right:
            act = 'a2'
        obj = ''
        if observation == Tiger_Left:
            obj = 'o0'
        elif observation == Tiger_Right:
            obj = 'o1'
        base_sequence = base_sequence + act + obj
        return base_sequence

    # producing a sequences of tests representations
    def generate_tests(self, num=100):
        if len(self.U_t_Name) == 0:
            for i in range(len(self.Actions)):
                for j in range(len(self.Observations)):
                    self.U_t_Name.append(self.generate_test(base_sequence='', action=self.Actions[i], observation=self.Observations[j]))
        count = len(self.U_t_Name)
        k=0
        while count < num:
            base_seq = self.U_t_Name[k]
            k=k+1
            for i in range(len(self.Actions)):
                for j in range(len(self.Observations)):
                    self.U_t_Name.append(self.generate_test(base_sequence=base_seq, action=self.Actions[i], observation=self.Observations[j]))
                    count = len(self.U_t_Name)
                    if count >= num:
                        return self.U_t_Name
        return self.U_t_Name

    # Computing the probabilities of tests
    def Computing_U_T(self):
        for i in range(len(self.U_t_Name)):
            U_t_Name = self.U_t_Name[i]
            U_t = np.identity(2)
            iters = np.arange(0, len(U_t_Name), 4)
            for j in iters:
                if U_t_Name[j] == 'a':
                    num_a = np.int(x=U_t_Name[j+1])
                    num_o = np.int(x=U_t_Name[j+3])
                    T = self.T[num_a]
                    O = self.O[num_a][num_o]
                    U_t = np.matmul(a=U_t, b=np.matmul(O, T))
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
        iters = np.arange(0, len(test_name), 4)
        test_pro = np.identity(2)
        for j in iters:
            if test_name[j] == 'a':
                num_a = np.int(x=test_name[j + 1])
                num_o = np.int(x=test_name[j + 3])
                T = self.T[num_a]
                O = self.O[num_a][num_o]
                test_pro = np.matmul(a=test_pro, b=np.matmul(O, T))
            else:
                print('exception on Computing tests')
        e = np.ones(shape=(np.shape(a=test_pro)[1], 1))
        test_pro = np.matmul(a=test_pro, b=e)
        return test_pro

    # obtain M_ao for a specific ao
    def gain_M_ao(self, ao):
        M_ao = []
        M_ao_name = []
        for i in range(len(self.U_Q_Name)):
            M_ao.append(solve(self.U_Q, self.gain_U_t(test_name=ao+self.U_Q_Name[i])))
            M_ao_name.append(self.U_Q_Name[i])
        M_ao = np.array(M_ao).T
        self.M.append(M_ao)
        self.M_name.append(ao)
        return [M_ao, M_ao_name]

    ### trouble on M_ao^T * M_ao^T * m_ao^T
    # def gain_m_t(self, test):
    #     iters = np.arange(0, len(test)-4, 4)
    #     m_test = 1
    #     for i in iters:
    #         ao = test[i:i+4]
    #         if ao not in self.M_name:
    #             m_test = np.dot(m_test, np.array(self.gain_M_ao(ao=ao).T))
    #         elif ao in self.M_name:
    #             m_test = np.dot(m_test, np.array(self.M[self.M_name.index(ao)]).T)
    #     ao = test[-4:]
    #     m_ao = self.m[self.m_name.index(ao)]
    #     m_ao = np.reshape(m_ao, newshape=(-1, 1))
    #     m_test = np.dot(m_test, m_ao)
    #     return m_test
    # updating h to h_ao
    def update(self, action_idx, observation, count):
        act = 'a' + str(action_idx)
        ob = 'o'
        if observation == Tiger_Left:
            ob = ob + '0'
        elif observation == Tiger_Right:
            ob = ob + '1'
        test_ao = act + ob
        if test_ao not in self.m_name:
            self.gain_m()
        index = self.m_name.index(test_ao)
        m_ao = self.m[index]
        m_ao = np.reshape(a=m_ao, newshape=(-1,))
        M_ao = None
        if test_ao not in self.M_name:
            M_ao, M_ao_name = self.gain_M_ao(test_ao)
        elif test_ao in self.M_name:
            M_ao = self.M[self.M_name.index(test_ao)]
        if self.Predictive_State is None:
            self.Predictive_State = self.return_predictive_state()
        denominator = np.matmul(a=self.Predictive_State.T, b=m_ao)
        numerator = np.matmul(a=self.Predictive_State.T, b=M_ao)
        new_Predictive_State = numerator/denominator
        self.Predictive_State = new_Predictive_State.T
        self.print_all(File_name=str(count)+test_ao)
        return self.Predictive_State

    # return predictive state P(Q|h)
    def return_predictive_state(self):
        U_Q = self.U_Q
        b_h = self.b_h
        return np.matmul(a=b_h, b=U_Q).T

    def gain_core_tests_dim(self):
        return np.prod(np.shape(self.Predictive_State), axis= -1)

    # print all details
    def print_all(self, File_name):
        U_Q = pd.DataFrame(data=self.U_Q, columns=self.U_Q_Name)
        M_ao = pd.DataFrame.from_records(data=self.M, index=self.M_name)
        m_ao = pd.DataFrame.from_records(data=self.m, index=self.m_name)
        U_Q.to_csv(path_or_buf=File_name+'.csv', mode='a')
        M_ao.to_csv(path_or_buf=File_name+'.csv', mode='a')
        m_ao.to_csv(path_or_buf=File_name+'.csv', mode='a')