import numpy as np
import matplotlib.pyplot as plt
from sklearn.QuantumUtility.Utility import *
import warnings
from fitter import *

warnings.filterwarnings("ignore")
# LOAD ARRAY

v_exp = np.load('array784_exp.npy')
v_uni = np.load('array784.npy')
v_sparse20 = np.load('sparse_arr_20.npy')
v_sparse50 = np.load('sparse_arr_50.npy')

# CREATE NEW ARRAY
# new_vec = np.array(create_rand_vec(1, 100))
list_ = [v_exp, v_uni, v_sparse20, v_sparse50]
delta = 0.9


# MAKE TOMOGRAPHY

def decreasing_error_plot(vector_list, delta):
    if len(vector_list) > 1:
        it = iter(vector_list)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')
    N = int((36 * len(vector_list[0]) * np.log(len(vector_list[0]))) / (delta ** 2))
    fig = plt.figure()
    for e, vector in enumerate(vector_list):
        if np.linalg.norm(vector) != 0.999 or np.linalg.norm(vector) != 1.0:
            vector = vector / np.linalg.norm(vector, ord=2)
        dictionary_estimates = L2_tomography(vector, delta=delta, stop_when_reached_accuracy=False,
                                                       sparsity_percentage=True)
        measure = list(dictionary_estimates.keys())
        samples = list(dictionary_estimates.values())
        samples_ = [np.linalg.norm(vector - samples[i], ord=2) for i in range(len(samples))]

        idx = np.argwhere(
            np.diff(np.sign(np.full(shape=len(samples_), fill_value=delta) - samples_))).flatten()

        plt.plot(measure[idx[0]], delta, 'ro')
        plt.plot(measure, samples_, label='vector' + str(e) + str((measure[idx[0]], delta)))

    plt.axhline(y=delta, color='k', linestyle='--', label='delta=' + str(delta))
    plt.axvline(x=N, color='k', linestyle='dotted', label='N =' + str(N))
    plt.xscale('log')
    plt.title("Error decrease for vectors of length"+ str(len(vector_list[0])),
              fontdict={'family': 'serif',
                        'color': 'darkblue',
                        'weight': 'bold',
                        'size': 18})
    plt.legend()
    plt.show()


def make_real_predicted_comparison(vector, delta):
    if np.linalg.norm(vector) != 0.999 or np.linalg.norm(vector) != 1.0:
        vector = vector / np.linalg.norm(vector, ord=2)
    dictionary_estimates = L2_tomography(vector, delta=delta, stop_when_reached_accuracy=False,
                                                   sparsity_percentage=True)
    measure = list(dictionary_estimates.keys())
    samples = list(dictionary_estimates.values())
    samples_ = [np.linalg.norm(vector - samples[i], ord=2) for i in range(len(samples))]
    idx = np.argwhere(
        np.diff(np.sign(np.full(shape=len(samples_), fill_value=delta) - samples_))).flatten()
    N = int((36 * len(vector) * np.log(len(vector))) / (delta ** 2))

    fig = plt.figure()

    plt.plot(measure[idx[0]], delta, 'ro')
    plt.plot(N, delta, 'ro')
    plt.title("Comparison between real and predicted measurements",
              fontdict={'family': 'serif',
                        'color': 'darkblue',
                        'weight': 'bold',
                        'size': 18})

    def compute_predicted_measure(len_vec, delta):
        return int((36 * len_vec * np.log(len_vec)) / (delta ** 2))

    predicted_points = [compute_predicted_measure(len(vector), i) for i in samples_]

    plt.axhline(y=delta, color='k', linestyle='--', label='delta=' + str(delta))
    plt.axvline(x=N, color='k', linestyle='dotted', label='N=' + str(N))
    plt.xscale('log')
    plt.plot(predicted_points, samples_, label="theory" + str((N, delta)))
    plt.plot(measure, samples_, label="real" + str((measure[idx[0]], delta)))
    plt.legend()
    plt.show()


def found_distribution(vector, n_measurements, delta, distribution_fitter=False, distributions=None):
    samples = []
    if np.linalg.norm(vector) != 0.999 or np.linalg.norm(vector) != 1.0:
        vector = vector / np.linalg.norm(vector, ord=2)

    for i in range(n_measurements):
        B = L2_tomography(vector, delta=delta, stop_when_reached_accuracy=False)
        print(i)
        # Append the Frobenius norm of A-B to the samples
        B = np.array(list(B.values())[-1])
        samples.append(np.linalg.norm(vector - B, ord=2))

    # Plot the samples
    plt.hist(samples, bins=50, color="darkblue")
    plt.xlabel(r"$||\mathbf{v} - \overline{\mathbf{v}}||_F$")
    plt.ylabel("measurements")

    if distribution_fitter:
        f = Fitter(samples, distributions=distributions, timeout=100)
        f.fit()
        f.summary()
        print(f.get_best(method='sumsquare_error'))
    plt.show()

#decreasing_error_plot(list_, delta)
#make_real_predicted_comparison(v_uni, delta)
#found_distribution(vector=v_uni, n_measurements=1000, delta=.9,distribution_fitter=True)