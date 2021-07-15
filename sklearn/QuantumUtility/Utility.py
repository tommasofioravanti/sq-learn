from collections import Counter
import math
import random
import re
import time
from scipy.stats import truncnorm
import numpy as np
from joblib import Parallel
from multiprocessing import *
import os
#random.seed(a=1234, version=2)

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
        #return random.choices(self.registers,weights=self.probabilities,k=n_times)
        LocalProcRandGen = np.random.RandomState()
        return LocalProcRandGen.choice(self.registers, p=self.probabilities, size=n_times)
        #return np.random.choice(self.registers, p=self.probabilities, size=n_times)

    def get_state(self):
        return {self.registers[i]: self.probabilities[i] for i in range(len(self.registers))}


def estimate_wald(measurements):
    counter = Counter(measurements)
    #keys = list(counter.keys())
    #values = list(np.asarray(list(counter.values()))/len(measurements))
    #estimate = dict(zip(keys,values))
    estimate = {x: counter[x] / len(measurements) for x in counter}
    return estimate

###[1 4 5 6 2 3 4 1 4 8 9 6] 12    [1 4 5] [6 2 3] [4 1 4] [8 9 6]

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

def make_noisy_vec(vec, noise,tomography=False):
    if tomography:
        tomography_res = L2_tomogrphy_parallel(vec, delta=noise,n_jobs=-1)
        new_vec = list(tomography_res.values())[-1] #take the last elements of the dictionary
    else:

        noise_per_component = noise / np.sqrt(len(vec))
        if noise_per_component != 0:

            errors = truncnorm.rvs(-noise_per_component, noise_per_component, size=len(vec))
            somma = lambda x, y: x + y
            # new_vec = np.array([vec[i] + errors[i] for i in range(len(vec))])
            new_vec = np.apply_along_axis(somma, 0, vec, errors)

        else:
            new_vec = vec

    return new_vec

#Given a matrix it makes it noisy by adding gaussian error to each component
def make_noisy_mat(A, noise, tomography=False):
    if tomography:
        vector_list = []
        for i in range(A.shape[0]):
            vector_list.append(make_noisy_vec(A[i], noise=noise, tomography=tomography))
            print(i)
        B = np.array(vector_list)
    else:


        vector_A = A.reshape(A.shape[0]*A.shape[1])
        vector_B = make_noisy_vec(vector_A, noise)
        B = vector_B.reshape(A.shape[0],A.shape[1])
    return B


def create_rand_vec(n_vec, len_vec,scale=None,type = 'uniform'):
    v=[]
    for i in range(n_vec):
        if type == 'uniform':
            vv = np.random.uniform(-1, 1, (len_vec))
        elif type == 'exp':
            vv = np.random.exponential(scale=scale,size=len_vec)
        #elif type =='int':
            #vv = np.random.randint(1, 100000000000, 784)
        vv = vv/np.linalg.norm(vv,ord=2)
        v.append(vv)

    return v


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


def L2_tomogrphy_parallel(V, N = None, delta=None,stop_when_reached_accuracy=True,n_jobs=None):

    if np.round(np.linalg.norm(V, ord=2)) == 1.0 or np.round(np.linalg.norm(V, ord=2)) == 0.9:
        pass
    else:

        V = V / np.linalg.norm(V, ord = 2)

    d = len(V)
    index = np.arange(0, d)
    if N is None:
        N = int((36 * d * np.log(d)) / (delta ** 2))

    q_state = QuantumState(amplitudes=V, registers=index)
    dict_res = {}

    if n_jobs == None:
        #Case not parallel
        return L2_tomographyVector_rightSign(V=V,delta=delta)
    elif n_jobs ==-1:
        n_cpu = cpu_count()
    else:
        n_cpu = n_jobs
        assert ( cpu_count() >= n_cpu)
    #block = int(N * frac)
    measure_indexes = np.geomspace(1, N, num=100, dtype=int)
    measure_indexes = check_measure(measure_indexes)

    for i in measure_indexes:
        if i < n_cpu:
            P = estimate_wald(q_state.measure(n_times=int(i)))
        else:
            with Pool(n_cpu) as prc:
                measure_lists = prc.starmap(auxiliary_fun, list(zip([q_state]*n_cpu, [int(i/n_cpu)]*n_cpu)))
                P_ = prc.map(estimate_wald,measure_lists)
            P_ = [Counter(c) for c in P_]
            P = sum(P_, Counter())
            P = {x: P[x] / n_cpu for x in P}
        P_i = np.zeros(d)

        P_i[list(P.keys())]=np.sqrt(list(P.values()))
        # Part2 of algorithm 4.1
        max_index = max(index)
        digits = len(str(max_index)) + 1
        registers = [str(j).zfill(digits) for j in index] + [re.sub('0', '1', str(j).zfill(digits), 1) for j in index]

        amplitudes = np.asarray([V[k] + P_i[k] for k in range(len(V))] + [V[k] - P_i[k] for k in range(len(V))])

        amplitudes *= 0.5

        new_quantum_state = QuantumState(registers=registers, amplitudes=amplitudes)
        if i < n_cpu:
            measure = new_quantum_state.measure(n_times=int(i))
        else:

            with Pool(n_cpu) as proc:
                measure = proc.starmap(auxiliary_fun, list(zip([new_quantum_state]*n_cpu, [int(i/n_cpu)]*n_cpu)))
                #measure = new_quantum_state.measure(int(i))  # ritorna una lista tipo ['01','02','02',...] lunga N
                measure = np.concatenate(measure)

        str_ = [str(ind).zfill(digits) for ind in index]
        dictionary = dict(Counter(measure))

        if len(dictionary) < len(registers):
            keys = set(list(dictionary.keys()))
            tot_keys = set(registers)
            missing_keys = list(tot_keys - keys)
            dictionary.update({l: 0 for l in missing_keys})

        d_ = list(map(dictionary.get, str_))

        P_i = [P_i[e] if x > 0.4 * P_i[e] ** 2 * i else P_i[e] * -1 for e, x in enumerate(d_)]
        print(i)
        #print(i, np.linalg.norm(P_i,ord=2))



        dict_res.update({i: P_i})
        if stop_when_reached_accuracy:

            sample = np.linalg.norm(V - P_i, ord=2)
            print(sample,i)
            if sample > delta:
                pass
            else:
                break


    return dict_res



def L2_tomographyVector_rightSign(V, N = None, delta=None):
    if np.round(np.linalg.norm(V, ord=2)) == 1.0 or np.round(np.linalg.norm(V, ord=2)) == 0.9:
        pass
    else:

        V = V / np.linalg.norm(V, ord = 2)
    d = len(V)
    index = np.arange(d)
    if N is None:
        N = (36*d*np.log(d)) / (delta**2)

    q_state = QuantumState(amplitudes=V, registers=index)


    n_cpu = cpu_count()
    with Pool(n_cpu) as prc:
        measure_lists = prc.starmap(auxiliary_fun, list(zip([q_state] * n_cpu, [int(N / n_cpu)] * n_cpu)))
        P_ = prc.map(estimate_wald, measure_lists)
    P_ = [Counter(c) for c in P_]
    P = sum(P_, Counter())
    P = {x: P[x] / n_cpu for x in P}


    P_i = np.zeros(d)
    P_i[list(P.keys())]=np.sqrt(list(P.values()))
    #P_i = list(np.vectorize(vectorize_aux_fun)(P,index))
    # Part2

    max_index = max(index)
    digits = len(str(max_index)) + 1
    registers = [str(i).zfill(digits) for i in index] +  [re.sub('0','1',str(i).zfill(digits),1) for i in index]

    amplitudes = np.asarray([V[i] + P_i [i] for i in range(len(V))] + [V[i] - P_i [i] for i in range(len(V))])

    ## V deve essere unitario
    amplitudes *= 0.5

    new_quantum_state = QuantumState(registers=registers , amplitudes=amplitudes)

    with Pool(n_cpu) as proc:
        measure = proc.starmap(auxiliary_fun, list(zip([new_quantum_state] * n_cpu, [int(N / n_cpu)] * n_cpu)))
    measure = np.concatenate(measure)

    #measure = new_quantum_state.measure(int(N))  #ritorna una lista tipo ['01','02','02',...] lunga N
    str_ = [str(ind).zfill(digits) for ind in index]
    dictionary = dict(Counter(measure))
    if len(dictionary) < len(registers):
        keys = set(list(dictionary.keys()))
        tot_keys = set(registers)

        missing_keys = list(tot_keys - keys)

        dictionary.update({l: 0 for l in missing_keys})

    d_ = list(map(dictionary.get, str_))

    P_i = [P_i[e] if i > 0.4 * P_i[e] ** 2 * N else P_i[e] * -1 for e, i in enumerate(d_)]


    return P_i


def L2_tomographyMatrix_rightSign(M, N = None, delta=None):
    V = M.reshape(M.shape[0] * M.shape[1])


    d = len(V)
    index = np.arange(d)
    if N is None:
        N = (36*d*np.log(d)) / (delta**2)

    q_state = QuantumState(amplitudes=V, registers=index)


    n_cpu = cpu_count()
    with Pool(n_cpu) as prc:
        measure_lists = prc.starmap(auxiliary_fun, list(zip([q_state] * n_cpu, [int(N / n_cpu)] * n_cpu)))
        P_ = prc.map(estimate_wald, measure_lists)
    P_ = [Counter(c) for c in P_]
    P = sum(P_, Counter())
    P = {x: P[x] / n_cpu for x in P}

    P_i = np.zeros(d)
    #P = estimate_wald(q_state.measure(n_times=int(N)))


    #Manage the mismatch problem of the length of the measurements for some values



    P_i[list(P.keys())]=list(P.values())

    # Part2

    max_index = max(index)
    digits = len(str(max_index)) + 1
    registers = [str(i).zfill(digits) for i in index] +  [re.sub('0','1',str(i).zfill(digits),1) for i in index]

    amplitudes = np.asarray([V[i] + P_i [i] for i in range(len(V))] + [V[i] - P_i [i] for i in range(len(V))])

    ## V deve essere unitario
    amplitudes *= 0.5

    new_quantum_state = QuantumState(registers=registers , amplitudes=amplitudes)

    with Pool(n_cpu) as proc:
        measure = proc.starmap(auxiliary_fun, list(zip([new_quantum_state] * n_cpu, [int(N / n_cpu)] * n_cpu)))
        # measure = new_quantum_state.measure(int(i))  # ritorna una lista tipo ['01','02','02',...] lunga N
    measure = np.concatenate(measure)

    #measure = new_quantum_state.measure(int(N))  #ritorna una lista tipo ['01','02','02',...] lunga N
    str_ = [str(ind).zfill(digits) for ind in index]
    dictionary = dict(Counter(measure))
    if len(dictionary) < len(registers):
        keys = set(list(dictionary.keys()))
        tot_keys = set(registers)

        missing_keys = list(tot_keys - keys)
        # v_set = set(V)
        # missing_values = list(v_set - keys)
        dictionary.update({l: 0 for l in missing_keys})

    d_ = list(map(dictionary.get, str_))

    P_i = np.asarray([P_i[e] if i > 0.4 * P_i[e] ** 2 * N else P_i[e] * -1 for e, i in enumerate(d_)])
    B = P_i.reshape(M.shape[0], M.shape[1])

    return B

def auxiliary_fun(q_state,i):
    P= q_state.measure(n_times=int(i))
    #print(os.getpid)
    return P
def vectorize_aux_fun(dic,i):
    return np.sqrt(dic[i]) if i in dic else 0

def check_measure(arr):
    for i in range(len(arr) - 1):
        if arr[i + 1] == arr[i]:
            arr[i + 1] += 1
        if arr[i + 1] <= arr[i]:
            arr[i + 1] = arr[i] + 1
    return arr


def L2_tomogrphy_faster(V, N = None, delta=None,frac=0.01 ,n_jobs=None):

    if np.round(np.linalg.norm(V, ord=2)) == 1.0 or np.round(np.linalg.norm(V, ord=2)) == 0.9:
        pass
    else:

        V = V / np.linalg.norm(V, ord = 2)

    d = len(V)
    index = np.arange(0, d)
    if N is None:
        N = int((36 * d * np.log(d)) / (delta ** 2))

    q_state = QuantumState(amplitudes=V, registers=index)
    dict_res = {}

    assert (frac > 0)
    if n_jobs == None:
        #Case not parallel
        return L2_tomographyVector_rightSign(V=V,delta=delta)
    elif n_jobs ==-1:
        n_cpu = cpu_count()
    else:
        n_cpu = n_jobs
        assert ( cpu_count() >= n_cpu)
    block = int(N * frac)
    with Pool(n_cpu) as prc:
        measure_lists = prc.starmap(auxiliary_fun, list(zip([q_state] * n_cpu, [int(N/ n_cpu)] * n_cpu)))
    measure_lists = np.concatenate(measure_lists, axis=0)
    counter = 0
    for i in range(block, N, block):
        start = time.time()
        counter+=1
        with Pool(n_cpu) as prc:

            measure_ =np.array_split(measure_lists[0:i+1], n_cpu)
            P_ = prc.map(estimate_wald,measure_)
        P_ = [Counter(c) for c in P_]
        P = sum(P_, Counter())
        P = {x: P[x] / n_cpu for x in P}
        P_i = np.zeros(d)

        P_i[list(P.keys())]=np.sqrt(list(P.values()))
        # Part2 of algorithm 4.1
        max_index = max(index)
        digits = len(str(max_index)) + 1
        registers = [str(j).zfill(digits) for j in index] + [re.sub('0', '1', str(j).zfill(digits), 1) for j in index]

        amplitudes = np.asarray([V[k] + P_i[k] for k in range(len(V))] + [V[k] - P_i[k] for k in range(len(V))])

        amplitudes *= 0.5

        new_quantum_state = QuantumState(registers=registers, amplitudes=amplitudes)
        if counter==1:

            with Pool(n_cpu) as proc:
                measure = proc.starmap(auxiliary_fun, list(zip([new_quantum_state]*n_cpu, [int(N/n_cpu)]*n_cpu)))
                #measure = new_quantum_state.measure(int(i))  # ritorna una lista tipo ['01','02','02',...] lunga N
                measure = np.concatenate(measure)


        str_ = [str(ind).zfill(digits) for ind in index]
        mea = measure[0:i+1]
        dictionary = dict(Counter(mea))

        if len(dictionary) < len(registers):
            keys = set(list(dictionary.keys()))
            tot_keys = set(registers)
            missing_keys = list(tot_keys - keys)
            dictionary.update({l: 0 for l in missing_keys})

        d_ = list(map(dictionary.get, str_))

        P_i = [P_i[e] if x > 0.4 * P_i[e] ** 2 * i else P_i[e] * -1 for e, x in enumerate(d_)]
        print(i)
        #print(i, np.linalg.norm(P_i,ord=2))

        dict_res.update({i: P_i})
        end = time.time()
        print(i, end-start)

    return dict_res
