from collections import Counter
import math
import random
from scipy.stats import truncnorm
import numpy as np
class QuantumState(object):
    """This class simulates a simple Quantum Register"""

    def __init__(self, registers, amplitudes):
        super(QuantumState, self).__init__()
        self.registers = registers

        # Amplitudes must be normalized to have the right probabilities for each register
        self.norm_factor = math.sqrt(sum([pow(x, 2) for x in amplitudes]))
        self.amplitudes = [x / self.norm_factor for x in amplitudes]

        # Each register_i appears with probability amplitude_i^2
        self.probabilities = [pow(x, 2) for x in self.amplitudes]
        assert (len(self.registers) == len(self.amplitudes))
        assert (abs(sum(self.probabilities) - 1) < 0.0000000001)

    def measure(self, n_times=1):
        return random.choices(self.registers, weights=self.probabilities, k=n_times)

    def get_state(self):
        return {self.registers[i]: self.probabilities[i] for i in range(len(self.registers))}


def estimate_wald(measurements):
    counter = Counter(measurements)
    estimate = {x: counter[x] / len(measurements) for x in counter}
    return estimate


## Introduce +-epsilon error in a value
def introduce_error(value, epsilon):
    return value + truncnorm.rvs(-epsilon, epsilon, size=1)


def coupon_collect(quantum_state):
    counter = 0
    collection_dict = {value: 0 for value in quantum_state.get_state().keys()}

    # Until you don't collect all the values, keep sampling and increment the counter
    while sum(collection_dict.values()) != len(collection_dict):
        value = quantum_state.measure()[0]
        if not collection_dict[value]:
            collection_dict[value] = 1
        counter += 1
    return counter

def make_noisy_vec(vec, noise,unitary=False):
    noise_per_component = noise/np.sqrt(len(vec))
    if noise_per_component != 0:
        errors = truncnorm.rvs(-noise_per_component,noise_per_component, size=len(vec))
        somma = lambda x, y: x + y
        #new_vec = np.array([vec[i] + errors[i] for i in range(len(vec))])
        new_vec = np.apply_along_axis(somma, 0, vec, errors)
        if unitary:
            #make the vector of unitary norm
            new_vec = new_vec / np.linalg.norm(new_vec, ord=2)
            print(np.linalg.norm(new_vec, ord=2))
    else:
        new_vec=vec
        #print(np.linalg.norm(new_vec,ord=2))
    return new_vec

#Given a matrix it makes it noisy by adding gaussian error to each component
def make_noisy_mat(A, noise, unitary=False):
    vector_A = A.reshape(A.shape[0]*A.shape[1])
    vector_B = make_noisy_vec(vector_A, noise,unitary)
    B = vector_B.reshape(A.shape[0],A.shape[1])
    return B

def L2_tomogrphy_fakeSign(V, N = None, delta=None):
    d = len(V)
    #index = np.arange(0,d)
    if N is None:
        N = (36*d*np.log(d)) / (delta**2)

    q_state = QuantumState(amplitudes=V, registers=V)
    P = estimate_wald(q_state.measure(n_times=int(N)))
    #P = {V[k]: v for (k, v) in P.items()}

    #Manage the mismatch problem of the length of the measurements for some values
    if len(P) < d:
        keys = set(list(P.keys()))
        v_set = set(V)
        missing_values = list(v_set - keys)
        P.update({l : 0 for l in missing_values})

    P_sqrt = {k : (-np.sqrt(v) if k<0 else np.sqrt(v)) for (k, v) in P.items()}

    x_sign=list(map(P_sqrt.get,V))
    #x_sign = np.array(list(P_sqrt.values()))
    #print(np.linalg.norm(x_sign,ord=2))
    return x_sign

def L2_tomogrphy_incremental(V, N = None, delta=None):
    d = len(V)
    #index = np.arange(0,d)
    if N is None:
        N = int((36*d*np.log(d)) / (delta**2))

    q_state = QuantumState(amplitudes=V, registers=V)
    dict_res={}
    for i in range(100000, N, 100000):
        P = estimate_wald(q_state.measure(n_times=int(i)))
    #P = {V[k]: v for (k, v) in P.items()}

    #Manage the mismatch problem of the length of the measurements for some values
        if len(P) < d:
            keys = set(list(P.keys()))
            v_set = set(V)
            missing_values = list(v_set - keys)
            P.update({l : 0 for l in missing_values})

        P_sqrt = {k : (-np.sqrt(v) if k<0 else np.sqrt(v)) for (k, v) in P.items()}

        x_sign=list(map(P_sqrt.get,V))
        #print(i)
        dict_res.update({i:x_sign})
    #x_sign = np.array(list(P_sqrt.values()))
    #print(np.linalg.norm(x_sign,ord=2))
    return dict_res

def L2_tomogrphy_rightSign(V, N = None, delta=None):
    d = len(V)
    index = np.arange(d)
    if N is None:
        N = (36*d*np.log(d)) / (delta**2)

    q_state = QuantumState(amplitudes=V, registers=index)
    P = estimate_wald(q_state.measure(n_times=int(N)))
    #P = {V[k]: v for (k, v) in P.items()}

    #Manage the mismatch problem of the length of the measurements for some values
    if len(P) < d:
        keys = set(list(P.keys()))
        tot_keys =set(list(index))
        missing_keys = list(tot_keys - keys)
        #v_set = set(V)
        #missing_values = list(v_set - keys)
        P.update({l : 0 for l in missing_keys})
    P_sqrt = {V[k]:np.sqrt(v) for (k, v) in P.items()}
    P_i = list(map(P_sqrt.get, V))

    max_index = max(index)
    digits = len(str(max_index)) + 1
    registers = [str(i).zfill(digits) for i in index] +  [str(i).rjust(digits, '1') for i in index]

    amplitudes = np.asarray([V[i] + P_i [i] for i in range(len(V))] + [V[i] - P_i [i] for i in range(len(V))])

    amplitudes *= 0.5

    new_quantum_state = QuantumState(registers=registers , amplitudes=amplitudes)

    ##TODO:: FARE MISURE DEL NUOVO STATO E VERIFICARE SEGNO TRAMITE FORMULINA






    #x_sign = np.array(list(P_sqrt.values()))
    #print(np.linalg.norm(x_sign,ord=2))
    return x_sign



def create_rand_vec(n_vec, len_vec,scale=None,type = 'uniform'):
    v=[]
    for i in range(n_vec):
        if type == 'uniform':
            vv = np.random.uniform(-1, 1, (len_vec))
        elif type == 'exp':
            vv = np.random.exponential(scale=scale,size=len_vec)
        vv = vv/np.linalg.norm(vv,ord=2)
        v.append(vv)
        print(np.linalg.norm(vv,ord=2))
    return v
