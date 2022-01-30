
import pandas as pd

from Models.PCAmodel import Model
from sklearn.decomposition import PCA, qPCA
from tabulate import tabulate

from sklearn.QuantumUtility.Utility import *
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../kddcup.data_10_percent_corrected', header=None)
df1 = df.drop(columns=[1, 2, 3, 6, 11, 20, 21])
df1.loc[df1[41] != 'normal.', 41] = 'attack'

trains = []
tests = []
trains_without_labels = []
labels = []
test = []
for i in range(5):
    trains.append(pd.read_csv('../KDDCUP/Trains/trains' + str(i)).drop(columns='Unnamed: 0'))
    tests.append(pd.read_csv('../KDDCUP/Tests/tests' + str(i)).set_index('Unnamed: 0'))

for i in range(5):
    x = tests[i].index
    test.append(df1.loc[x])
    labels.append(df1.loc[x][41])
    trains_without_labels.append(trains[i].drop(columns='41'))
# Compute Quantile---> 1-\alpha1
alpha = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
quantils = []
for i in alpha:
    eq = [1, -2, i]
    quantile = 1 - np.round(np.roots(eq)[1], decimals=4)
    quantils.append(quantile)

qpca30 = qPCA(svd_solver='full', name='qpca30').fit(trains[0].drop(columns='41'), theta_estimate=False,
                                                    estimate_all=True,
                                                    delta=0.1, eps=3, true_tomography=True,
                                                    stop_when_reached_accuracy=True,
                                                    theta=1)
print('PCA30 done', 'n_components', qpca30.topk)

qpca40 = qPCA(svd_solver='full', name='qpca40').fit(trains[1].drop(columns='41'), theta_estimate=False,
                                                    estimate_all=True,
                                                    delta=0.1, eps=3, true_tomography=True,
                                                    stop_when_reached_accuracy=True,
                                                    theta=1)
print('PCA40 done', 'n_components', qpca40.topk)
while True:
    try:
        qpca50 = qPCA(svd_solver='full', name='qpca50').fit(trains[2].drop(columns='41'), theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=True,
                                                            theta=1)
    except:
        pass
    else:
        break

print('PCA50 done', 'n_components', qpca50.topk)
while True:
    try:

        qpca60 = qPCA(svd_solver='full', name='qpca60').fit(trains[3].drop(columns='41'), theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=True,
                                                            theta=1)
    except:
        pass
    else:
        break
print('PCA60 done', 'n_components', qpca60.topk)
while True:
    try:
        qpca70 = qPCA(svd_solver='full', name='qpca70').fit(trains[4].drop(columns='41'), theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=True, theta=1)
    except:
        pass
    else:
        break
print('PCA70 done', 'n_components', qpca70.topk)
QPCAs = [qpca30, qpca40, qpca50, qpca60, qpca70]
#QPCAs = [qpca70]

qmodel = Model(QPCAs, quantils, quantum=True).fit(trains_without_labels, minor_sv_variance=0.20, only_dot_product=True,experiment=1)

recall1, precision1, accuracy1, f1_score1 = qmodel.predict(tests, labels, name_negative_class='normal.',
                                                           only_dot_product=True, experiment=1)

components = [i for i in recall1]
headers = ['FAR', 'PCC']
one_perc = [recall1[i][0] for i in components]
two_perc = [recall1[i][1] for i in components]
four_perc = [recall1[i][2] for i in components]
six_perc = [recall1[i][3] for i in components]
eight_perc = [recall1[i][4] for i in components]
ten_perc = [recall1[i][5] for i in components]

one_perc_avg = np.mean(one_perc)
one_perc_std = np.std(one_perc)

two_perc_avg = np.mean(two_perc)
two_perc_std = np.std(two_perc)

four_perc_avg = np.mean(four_perc)
four_perc_std = np.std(four_perc)

six_perc_avg = np.mean(six_perc)
six_perc_std = np.std(six_perc)

eight_perc_avg = np.mean(eight_perc)
eight_perc_std = np.std(eight_perc)

ten_perc_avg = np.mean(ten_perc)
ten_perc_std = np.std(ten_perc)

one_str = [str(one_perc_avg) + ' +/- ' + str(one_perc_std)]
two_str = [str(two_perc_avg) + ' +/- ' + str(two_perc_std)]
four_str = [str(four_perc_avg) + ' +/- ' + str(four_perc_std)]
six_str = [str(six_perc_avg) + ' +/- ' + str(six_perc_std)]
eight_str = [str(eight_perc_avg) + ' +/- ' + str(eight_perc_std)]
ten_str = [str(ten_perc_avg) + ' +/- ' + str(ten_perc_std)]

one_str.insert(0, '1%')
two_str.insert(0, '2%')
four_str.insert(0, '4%')
six_str.insert(0, '6%')
eight_str.insert(0, '8%')
ten_str.insert(0, '10%')

one_perc_p = [precision1[i][0] for i in components]
two_perc_p = [precision1[i][1] for i in components]
four_perc_p = [precision1[i][2] for i in components]
six_perc_p = [precision1[i][3] for i in components]
eight_perc_p = [precision1[i][4] for i in components]
ten_perc_p = [precision1[i][5] for i in components]

one_perc_avg_p = np.mean(one_perc_p)
one_perc_std_p = np.std(one_perc_p)

two_perc_avg_p = np.mean(two_perc_p)
two_perc_std_p = np.std(two_perc_p)

four_perc_avg_p = np.mean(four_perc_p)
four_perc_std_p = np.std(four_perc_p)

six_perc_avg_p = np.mean(six_perc_p)
six_perc_std_p = np.std(six_perc_p)

eight_perc_avg_p = np.mean(eight_perc_p)
eight_perc_std_p = np.std(eight_perc_p)

ten_perc_avg_p = np.mean(ten_perc_p)
ten_perc_std_p = np.std(ten_perc_p)

one_str_p = [str(one_perc_avg_p) + ' +/- ' + str(one_perc_std_p)]
two_str_p = [str(two_perc_avg_p) + ' +/- ' + str(two_perc_std_p)]
four_str_p = [str(four_perc_avg_p) + ' +/- ' + str(four_perc_std_p)]
six_str_p = [str(six_perc_avg_p) + ' +/- ' + str(six_perc_std_p)]
eight_str_p = [str(eight_perc_avg_p) + ' +/- ' + str(eight_perc_std_p)]
ten_str_p = [str(ten_perc_avg_p) + ' +/- ' + str(ten_perc_std_p)]

one_str_p.insert(0, '1%')
two_str_p.insert(0, '2%')
four_str_p.insert(0, '4%')
six_str_p.insert(0, '6%')
eight_str_p.insert(0, '8%')
ten_str_p.insert(0, '10%')

print('Detection Rate')
print(tabulate([one_str, two_str, four_str, six_str, eight_str, ten_str], headers=headers))
print("\n \n ")
print('Precision')
print(tabulate([one_str_p, two_str_p, four_str_p, six_str_p, eight_str_p, ten_str_p], headers=headers))

#qpca70.runtime_comparison(100000, 5000, 'KDDCUPExp1.pdf', estimate_components='right_sv', classic_runtime='classic')
qpca70.runtime_comparison(10000, 10, 'KDDCUPExp1zoom.pdf', estimate_components='right_sv', classic_runtime='classic')
