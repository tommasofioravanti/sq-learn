from collections import Counter
import math
import random
import re
import time
from scipy.stats import truncnorm
import numpy as np
import decimal
from joblib import Parallel
import statistics
import warnings
import matplotlib.pyplot as plt
from multiprocessing import *
import os


# random.seed(a=1234, version=2)

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
        # TODO: evaluate if there is benefits to move to scipy sampling rountine
        # return random.choices(self.registers, weights=self.probabilities, k=n_times)
        LocalProcRandGen = np.random.RandomState()
        return LocalProcRandGen.choice(self.registers, p=self.probabilities, size=n_times)
        # return np.random.choice(self.registers, p=self.probabilities, size=n_times)

    def get_state(self):
        return {self.registers[i]: self.probabilities[i] for i in range(len(self.registers))}


def estimate_wald(measurements):
    counter = Counter(measurements)
    # keys = list(counter.keys())
    # values = list(np.asarray(list(counter.values()))/len(measurements))
    # estimate = dict(zip(keys,values))
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


def make_gaussian_est(vec, noise):
    """This function is used to estimate vec with tomography or with Gaussian Noise Approximation.
    Parameters
    ----------
    vec : array-like that has to be estimated.

    noise : float. It represent the error that you want to introduce to estimate the representation of vector V.

    """
    noise_per_component = noise / np.sqrt(len(vec))
    if noise_per_component != 0:
        errors = truncnorm.rvs(-noise_per_component, noise_per_component, size=len(vec))
        print('noise_per_comp:', noise_per_component, 'errors:', errors)
        somma = lambda x, y: x + y
        # new_vec = np.array([vec[i] + errors[i] for i in range(len(vec))])
        new_vec = np.apply_along_axis(somma, 0, vec, errors)
    return new_vec


def tomography(A, noise, true_tomography=True, stop_when_reached_accuracy=True, N=None, norm='L2'):
    """ Tomography function (real and fake).
            Parameters
            ----------
            A : array-like that has to be estimated.

            N : int value, default=None.
                Number of measures of the quantum state. If None it is computed in the function itself.

            noise: float value.
                It represent the error that you want to introduce to estimate the representation of vector V.

            true_tomography: bool, default=True.
                If true means that the quantum estimations are are done with real tomography,
                otherwise the estimations are approximated with a Truncated Gaussian Noise.

            stop_when_reached_accuracy: bool value, default=True.
                If True it stops the execution of the tomography when the L2-norm of the
                difference between V and its estimation is less or equal then delta. Otherwise
                N measures are done (very memory intensive for large vectors).

            norm: string value, default='L2'
                If true_tomography is True:
                    'L2': L2-tomography is computed
                    'inf':L-inf tomography is computed

            Returns
            -------
            A_est : array-like that was estimated.

            Notes
            -----
            This method returns an estimation of the true array/matrix A using quantum tomography algorithm 4.1 proposed in
            "A Quantum Interior Point Method for LPs and SDPs" paper, or using an approximation of the tomography.
        """

    assert noise >= 0

    if noise == 0:
        return A

    if true_tomography == False:
        if len(A.shape) == 2:
            vector_A = A.reshape(A.shape[0] * A.shape[1])
            vector_B = make_gaussian_est(vector_A, noise)
            A_est = vector_B.reshape(A.shape[0], A.shape[1])
        else:
            A_est = make_gaussian_est(A, noise)

    else:
        if len(A.shape) == 2:
            A_est = np.array([np.array(list(
                real_tomography(A[idx], delta=noise, stop_when_reached_accuracy=stop_when_reached_accuracy,
                                N=N, norm=norm).values())[-1])
                              for idx in range(len(A))])
        else:
            A_est = np.array(
                list(real_tomography(A, delta=noise, stop_when_reached_accuracy=stop_when_reached_accuracy, N=N,
                                     norm=norm).values())[-1])

    return A_est


def create_rand_vec(n_vec, len_vec, scale=None, type='uniform'):
    v = []
    for i in range(n_vec):
        if type == 'uniform':
            vv = np.random.uniform(-1, 1, (len_vec))
        elif type == 'exp':
            vv = np.random.exponential(scale=scale, size=len_vec)
        vv = vv / np.linalg.norm(vv, ord=2)
        v.append(vv)

    return v


def L2_tomogrphy_fakeSign(V, N=None, delta=None):
    d = len(V)
    # index = np.arange(0,d)
    if N is None:
        N = (36 * d * np.log(d)) / (delta ** 2)

    q_state = QuantumState(amplitudes=V, registers=V)
    P = estimate_wald(q_state.measure(n_times=int(N)))
    # P = {V[k]: v for (k, v) in P.items()}

    # Manage the mismatch problem of the length of the measurements for some values
    if len(P) < d:
        keys = set(list(P.keys()))
        v_set = set(V)
        missing_values = list(v_set - keys)
        P.update({l: 0 for l in missing_values})

    P_sqrt = {k: (-np.sqrt(v) if k < 0 else np.sqrt(v)) for (k, v) in P.items()}

    x_sign = list(map(P_sqrt.get, V))
    # x_sign = np.array(list(P_sqrt.values()))
    # print(np.linalg.norm(x_sign,ord=2))
    return x_sign


def L2_tomogrphy_parallel(V, N=None, delta=None, stop_when_reached_accuracy=True, n_jobs=-1):
    if np.round(np.linalg.norm(V, ord=2)) == 1.0 or np.round(np.linalg.norm(V, ord=2)) == 0.9:
        pass
    else:

        V = V / np.linalg.norm(V, ord=2)

    d = len(V)
    index = np.arange(0, d)
    if N is None:
        N = int((36 * d * np.log(d)) / (delta ** 2))

    q_state = QuantumState(amplitudes=V, registers=index)
    dict_res = {}

    if n_jobs == None:
        # Case not parallel
        return real_tomography(V=V, delta=delta)
    elif n_jobs == -1:
        n_cpu = cpu_count()
    else:
        n_cpu = n_jobs
        assert (cpu_count() >= n_cpu)
    measure_indexes = np.geomspace(1, N, num=100, dtype=np.int64)

    measure_indexes = check_measure(measure_indexes)

    for i in measure_indexes:
        print(i)
        start = time.time()
        if i < n_cpu:
            P = estimate_wald(q_state.measure(n_times=int(i)))
        else:
            ll = check_division(i, n_cpu)
            with Pool(n_cpu) as prc:
                measure_lists = prc.starmap(auxiliary_fun, list(zip([q_state] * n_cpu, ll)))
                P_ = prc.map(estimate_wald, measure_lists)
            P_ = [Counter(c) for c in P_]
            P = sum(P_, Counter())
            P = {x: P[x] / n_cpu for x in P}
        P_i = np.zeros(d)

        P_i[list(P.keys())] = np.sqrt(list(P.values()))
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
                measure = proc.starmap(auxiliary_fun, list(zip([new_quantum_state] * n_cpu, ll)))
                # measure = new_quantum_state.measure(int(i))  # ritorna una lista tipo ['01','02','02',...] lunga N
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
        # print(i)
        # print(i, np.linalg.norm(P_i,ord=2))
        end = time.time()
        print('time:', end - start)
        dict_res.update({i: P_i})
        if stop_when_reached_accuracy:

            sample = np.linalg.norm(V - P_i, ord=2)
            # print(sample, i)
            if sample > delta:
                pass
            else:
                break

    return dict_res


def real_tomography(V, N=None, delta=None, stop_when_reached_accuracy=True, norm='L2', sparsity_percentage=False):
    """ Official version of the tomography function.
        Parameters
        ----------
        V : array-like that has to be estimated.

        N : int value, default=None.
            Number of measures of the quantum state. If None it is computed in the function itself.

        delta: float value, default=None.
             It represent the error that you want to introduce to estimate the representation of vector V.

        stop_when_reached_accuracy: bool, default=True.
                                    If True it stops the execution of the tomography when the L2-norm of the
                                    difference between V and its estimation is less or equal then delta. Otherwise
                                    N measures are done (very memory intensive for large vectors).
        sparsity_percentage: bool, default=False.
                            If True it computes the sparsity percentage of the vector and it is used to
                            make bigger and bigger measures as you increase the non-sparsity of the vector.
                            If False it makes measures that increases always of a factor of 1000.
                            It is particulary useful in testing the tomography.
        Returns
        -------
        dict_res : dictionary of shape {N_measure: vector_estimation}.

        Notes
        -----
        This method returns an estimation of the true array V using quantum tomography algorithm 4.1 proposed in
        "A Quantum Interior Point Method for LPs and SDPs" paper.

    """

    if sparsity_percentage:
        sparsity = 1 - (np.count_nonzero(V) / float(V.size))
    else:
        sparsity = None
    if np.round(np.linalg.norm(V, ord=2)) == 1.0 or np.round(np.linalg.norm(V, ord=2)) == 0.9:
        pass
    else:
        V = V / np.linalg.norm(V, ord=2)
    d = len(V)
    index = np.arange(0, d)
    if N is None:
        if norm == 'L2':
            N = int((36 * d * np.log(d)) / (delta ** 2))
        elif norm == 'inf':
            N = int((36 * np.log(d)) / (delta ** 2))

    q_state = QuantumState(amplitudes=V, registers=index)
    dict_res = {}

    measure_indexes = np.geomspace(1, N, num=100, dtype=np.int64)

    measure_indexes = check_measure(measure_indexes, sparsity, norm)

    for i in measure_indexes:

        P = estimate_wald(q_state.measure(n_times=int(i)))
        P_i = np.zeros(d)
        P_i[list(P.keys())] = np.sqrt(list(P.values()))

        # Part2 of algorithm 4.1
        max_index = max(index)
        digits = len(str(max_index)) + 1
        registers = [str(j).zfill(digits) for j in index] + [re.sub('0', '1', str(j).zfill(digits), 1) for j in index]

        amplitudes = np.asarray([V[k] + P_i[k] for k in range(len(V))] + [V[k] - P_i[k] for k in range(len(V))])

        amplitudes *= 0.5

        new_quantum_state = QuantumState(registers=registers, amplitudes=amplitudes)

        measure = new_quantum_state.measure(n_times=int(i))

        str_ = [str(ind).zfill(digits) for ind in index]
        dictionary = dict(Counter(measure))

        if len(dictionary) < len(registers):
            keys = set(list(dictionary.keys()))
            tot_keys = set(registers)
            missing_keys = list(tot_keys - keys)
            dictionary.update({l: 0 for l in missing_keys})

        d_ = list(map(dictionary.get, str_))

        P_i = [P_i[e] if x > 0.4 * P_i[e] ** 2 * i else P_i[e] * -1 for e, x in enumerate(d_)]

        dict_res.update({i: P_i})
        if stop_when_reached_accuracy:
            if norm == 'L2':
                sample = np.linalg.norm(V - P_i, ord=2)
            elif norm == 'inf':
                sample = np.linalg.norm(V - P_i, ord=np.inf)
            if sample > delta:
                pass
            else:
                break
    return dict_res


def auxiliary_fun(q_state, i):
    P = q_state.measure(n_times=int(i))
    return P


def vectorize_aux_fun(dic, i):
    return np.sqrt(dic[i]) if i in dic else 0


def check_measure(arr, sparsity, norm):
    if sparsity != None:
        incr = (1 - sparsity) * 1000 * 0.3 ** 2
    else:
        incr = 1000
    if norm == 'inf':
        incr = 5

    for i in range(len(arr) - 1):
        if arr[i + 1] == arr[i]:
            arr[i + 1] += incr
        if arr[i + 1] <= arr[i]:
            arr[i + 1] = arr[i] + incr
    return arr


def check_division(v, n_jobs):
    a = float(v) / n_jobs
    d = a - int(a)
    remaining = int(d * n_jobs)
    process_values = [int(a) for _ in range(n_jobs)]
    for i in range(remaining):
        process_values[i] += 1
    return process_values


def AmplitudeAmpDist(w0, w1):
    c = -np.ceil(w1 - w0)
    f = -np.floor(w1 - w0)
    distance = min(np.abs(c + w1 - w0), np.abs(f + w1 - w0))
    return distance


def amplitude_estimation(theta, epsilon, M=None, nqubit=False, plot_distribution=False):
    """ Official version of the amplitude estimation function.
        Parameters
        ----------
        theta: float or int value.
             Value that has to be estimated by the amplitude estimation. It must be in the range of [0,1].

        epsilon: float value.
            Error that you want to insert in the amplitude estimation procedure.

        nqubit: bool value, default=False.
            If True, the routine returns also the number of qubits to represent an estimate with the specified precision
            and the parameter M (see the amplitude estimation routine description for more details).

        plot_distribution: bool value, default=False.
            If True, a plot of the probability distribution for the output of amplitude estimation is done.


        Returns
        -------
        theta_tilde : float value.
            Estimation of the true theta.

        Notes
        -----
        This method performs the amplitude estimation routine following the approach described in "Quantum Amplitude Amplification
        and Estimation" paper. Amplitude estimation is the problem of estimating the probability that a measurements of
        a quantum state yields a good state.


    """
    if M == None:
        M = (math.ceil((np.pi / (2 * epsilon)) * (1 + np.sqrt(1 + 4 * epsilon))))
    else:
        warnings.warn(
            "Attention! The value of M that will be considered is the one you passed. Epsilon in this case is "
            "useless")

    p = []
    theta_j = []

    for j in range(M):
        theta_est = np.pi * j / M
        theta_j.append(theta_est / np.pi)
        distance = AmplitudeAmpDist(theta_est / np.pi, theta)
        if distance != 0:
            p_aj = np.abs(math.sin(M * distance * np.pi) / (M * math.sin(distance * np.pi))) ** 2
        else:
            p_aj = 1
        p.append(p_aj)
    # sum_at_one = np.sum(p)
    theta_tilde = random.choices(theta_j, weights=p, k=1)[0]

    n_qubits = np.log2(M)  # TODO: Check this value!

    if plot_distribution:
        plt.annotate((float('%.2f' % (theta_j[p.index(max(p))])), float('%.2f' % (max(p)))),
                     xy=(theta_j[p.index(max(p))], max(p)))
        plt.bar(theta_j, p, 0.001)

        plt.xlim(theta_j[p.index(max(p))] - 0.03, theta_j[p.index(max(p))] + 0.03)
        plt.title(r'Probability distribution for the output of amplitude estimation for $\theta$ = ' + str(
            float('%.2f' % (theta))) + r' with $\epsilon$ =' + str(epsilon) + r'$\rightarrow$ M=' + str(M),
                  fontdict={'family': 'serif',
                            'color': 'darkblue',
                            'weight': 'bold',
                            'size': 8})
        plt.xlabel(r'$\hat \theta$')
        plt.ylabel("probability")

        plt.show()

    a_tilde = math.cos(theta_tilde * np.pi / 2)
    if nqubit:
        return a_tilde, n_qubits, M
    return a_tilde


def median_evaluation(func, Q=None):
    # TODO
    if Q == None:
        #TODO:Formula of Q in NN
        Q

    estimates = [func for _ in Q]
    final_estimate = np.median(estimates)

    return final_estimate


def Wrapper_AmpEst(argument, type='singular_values'):
    if type == 'singular_values':
        theta_i = 2 * math.acos(argument)
        return theta_i


def compute_n_from_epsilon(epsilon):
    # TODO
    return


def brassard_amp_est(theta, M):
    # omega must be between [0,1]
    p = []
    a_j = []
    theta_j = []

    for j in range(0, M):
        theta_est = np.pi * j / M
        # a_j_current = (math.sin(theta_est)) ** 2
        theta_j.append(theta_est / np.pi)
        # a_j.append(a_j_current)

        # Se omega è tra [0,1] dividere theta_est per \pi
        distance = AmplitudeAmpDist(theta_est / np.pi, theta)
        if distance != 0:
            # In NN manca il *np.pi dentro il sin
            p_aj = np.abs((math.sin(M * distance * np.pi)) / (M * (math.sin(distance * np.pi)))) ** 2
        else:
            p_aj = 1
        p.append(p_aj)
    sum_at_one = np.sum(p)
    theta_tilde = random.choices(theta_j, weights=p, k=1)[0]

    # a_tilde = (math.sin(theta_tilde)) ** 2

    a_tilde = math.cos(theta_tilde * np.pi / 2)

    plt.annotate((float('%.2f' % (theta_j[p.index(max(p))])), float('%.2f' % (max(p)))),
                 xy=(theta_j[p.index(max(p))], max(p)))
    plt.bar(theta_j, p, 0.01)
    plt.title(r'Probability distribution for the output of amplitude estimation for $\theta$ = ' + str(
        float('%.2f' % (theta))),
              fontdict={'family': 'serif',
                        'color': 'darkblue',
                        'weight': 'bold',
                        'size': 10})
    plt.xlabel(r'$\hat \theta$')
    plt.ylabel("probability")

    plt.show()
    return a_tilde
