import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def quantils():
    alpha = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    quantils = []
    for i in alpha:
        eq = [1, -2, i]
        quantile = 1 - np.round(np.roots(eq)[1], decimals=4)
        quantils.append(quantile)
    return quantils


def loadDatasets():
    """ Load the 5 different Train/test already trimmed.
            Returns
            -------
            trains : List of 5 different Training set
            tests: List of 5 different Test set
            test: List of the Test set with labels (used for classification purposes).

        """

    df = pd.read_csv('kddcup.data_10_percent_corrected', header=None)
    df1 = df.drop(columns=[1, 2, 3, 6, 11, 20, 21])
    df1.loc[df1[41] != 'normal.', 41] = 'attack'

    cwd = os.getcwd()
    trains = []
    tests = []
    for filename in os.listdir(os.path.join(cwd, 'KDDCUP/Trains/')):
        trains.append(pd.read_csv(filename).drop(columns='Unnamed: 0'))

    for filename in os.listdir(os.path.join(cwd, 'KDDCUP/Tests/')):
        tests.append(pd.read_csv(filename).set_index('Unnamed: 0'))
    test = []
    for i in range(5):
        x = tests[i].index
        test.append(df1.loc[x])

    return trains, tests, test


def computeThreshold(QPCAs, trains):
    dictionary_major = {}
    quantils_ = quantils()
    for e, pca in enumerate(QPCAs):

        out_threshold_list_major = []
        emp_distribution_major = []

        for j in range(len(trains[e].drop(columns='41'))):
            sample = np.array(trains[e].drop(columns='41').iloc[j])
            y = np.dot(sample, pca.estimate_right_sv.T)
            s_major = np.sum(y ** 2 / pca.estimate_fs)

            emp_distribution_major.append(s_major)

        for q in quantils_:
            n_major = len(emp_distribution_major)

            sort_major = sorted(emp_distribution_major)

            out_threshold_major = sort_major[int(n_major * q)]

            out_threshold_list_major.append(out_threshold_major)

        dictionary_major.update({pca: out_threshold_list_major})

    return dictionary_major


def computeClassificationResult(QPCAs, trains, tests, test):
    dictionary_major = computeThreshold(QPCAs, trains=trains)
    classification_res = {}
    for e, pca in enumerate(QPCAs):

        acc_list = []
        for threshold_major in dictionary_major[pca]:
            print(threshold_major)
            TP = []
            FP = []
            TN = []
            FN = []

            for j in range(len(tests[e])):
                sample = np.array(tests[e].iloc[j])

                y = np.dot(sample, pca.estimate_right_sv.T)

                sum_major = np.sum(y ** 2 / pca.estimate_fs)

                if sum_major > threshold_major:
                    if test[e].iloc[j][41] == 'attack':
                        TP.append(j)
                    else:
                        FP.append(j)
                else:
                    if sum_major <= threshold_major:

                        if test[e].iloc[j][41] == 'attack':
                            FN.append(j)
                        else:
                            TN.append(j)

            accuracy = (len(TP)) / (len(TP) + len(FN))
            print(accuracy)
            acc_list.append(accuracy)

        classification_res.update({pca: acc_list})

    components = [i for i in classification_res]

    headers = ['FAR', 'PCA30', 'PCA40', 'PCA50', 'PCA60', 'PCA70']
    one_perc = [classification_res[i][0] for i in components]
    two_perc = [classification_res[i][1] for i in components]
    four_perc = [classification_res[i][2] for i in components]
    six_perc = [classification_res[i][3] for i in components]
    eight_perc = [classification_res[i][4] for i in components]
    ten_perc = [classification_res[i][5] for i in components]

    one_perc.insert(0, '1%')
    two_perc.insert(0, '2%')
    four_perc.insert(0, '4%')
    six_perc.insert(0, '6%')
    eight_perc.insert(0, '8%')
    ten_perc.insert(0, '10%')

    print(tabulate([one_perc, two_perc, four_perc, six_perc, eight_perc, ten_perc], headers=headers))


def y_classic(n, m):
    return n * m ** 2


def y_quantum(cost, n, m, add_cost):
    return cost * n * np.log2(n) * np.log2(n * m) + add_cost


def RunTimeEstimation(QPCAs, trains):
    # Computing other run-time parameters for matrix A

    for e, qpca in enumerate(QPCAs):
        a = trains[e].drop(columns='41') / qpca.spectral_norm
        max_ = 0
        for i in range(len(a)):
            l1_norm = np.sum([abs(a_) for a_ in a.loc[i]])
            if l1_norm > max_:
                max_ = l1_norm

        p = qpca.p
        one_over_theta = 1 / qpca.est_theta
        one_over_sqrtp = 1 / np.sqrt(p)
        muA_over_eps = max_ / qpca.eps
        k_over_delta2 = len(qpca.estimate_fs) / (qpca.delta) ** 2
        # k_classic = qpca.components_retained_
        cost = one_over_theta * one_over_sqrtp * muA_over_eps * k_over_delta2
        add_cost = max_ / qpca.eps_theta
        m = np.array([34, 50, 100, 150, 200, 300, 500], dtype="int64")
        n = np.array([5000, 7000, 7500, 8000, 10000, 20000, 25000], dtype="int64")
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        z = y_quantum(cost, n, m, add_cost)
        z1 = y_classic(n, m)

        ax.plot(n, m, z, label='quantum', marker='o')
        ax.plot(n, m, z1, label='classic', marker='o')
        plt.xlabel('n_samples')
        plt.ylabel('n_features')
        ax.legend()
        plt.title(qpca.name)
        plt.savefig(qpca.name)
