import pandas as pd

from Models.PCAmodel import Model
from sklearn.decomposition import PCA, qPCA
from tabulate import tabulate
import os
from sklearn.QuantumUtility.Utility import *
import warnings

warnings.filterwarnings("ignore")

cwd = os.getcwd()
tr = []
test = []
te = []
for filename in os.listdir(os.path.join(cwd, '../CICIDS2017/SavedTrainTest1')):
    if filename.startswith('train'):
        tr.append(pd.read_csv(os.path.join(cwd, '../CICIDS2017/SavedTrainTest1', filename)))
    elif filename.startswith('test'):
        test.append(pd.read_csv(os.path.join(cwd, '../CICIDS2017/SavedTrainTest1', filename)))
    else:
        te.append(pd.read_csv(os.path.join(cwd, '../CICIDS2017/SavedTrainTest1', filename)))
labels = []
for i in range(5):
    labels.append(test[i]['Label'])

for i in range(5):
    te[i] = te[i][:-50000]
    labels[i] = labels[i][:-50000]

#alpha = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
alpha = [0.01, 0.05, 0.07, 0.09, 0.1, 0.20]
quantils = []
for i in alpha:
    eq = [1, -2, i]
    quantile = 1 - np.round(np.roots(eq)[1], decimals=4)
    quantils.append(quantile)

qpca30 = qPCA(svd_solver='full', name='qpca30').fit(tr[0], theta_estimate=False,
                                                    estimate_all=True,
                                                    delta=0.1, eps=3, true_tomography=True,
                                                    stop_when_reached_accuracy=False,
                                                    theta=1)
print('PCA30 done', 'n_components', qpca30.topk)

qpca40 = qPCA(svd_solver='full', name='qpca40').fit(tr[1], theta_estimate=False,
                                                    estimate_all=True,
                                                    delta=0.1, eps=3, true_tomography=True,
                                                    stop_when_reached_accuracy=False,
                                                    theta=1)
print('PCA40 done', 'n_components', qpca40.topk)
while True:
    try:
        qpca50 = qPCA(svd_solver='full', name='qpca50').fit(tr[2], theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=False,
                                                            theta=1)
    except:
        pass
    else:

        break

print('PCA50 done', 'n_components', qpca50.topk)
while True:
    try:

        qpca60 = qPCA(svd_solver='full', name='qpca60').fit(tr[3], theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=False,
                                                            theta=1)
    except:
        pass
    else:
        break
print('PCA60 done', 'n_components', qpca60.topk)
while True:
    try:
        qpca70 = qPCA(svd_solver='full', name='qpca70').fit(tr[4], theta_estimate=False,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True,
                                                            stop_when_reached_accuracy=False, theta=1)
    except:
        pass
    else:
        break
print('PCA70 done', 'n_components', qpca70.topk)
QPCAs = [qpca30, qpca40, qpca50, qpca60, qpca70]
#QPCAs = [qpca70]

qmodel = Model(QPCAs, quantils, quantum=True).fit(tr, minor_sv_variance=0.20, only_dot_product=False,experiment=1)

recall1, precision1, accuracy1, f1_score1 = qmodel.predict(te, labels, name_negative_class='BENIGN',
                                                           only_dot_product=False, experiment=1)
AVG = 0
components = [i for i in recall1]
if AVG == 1:

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
    two_str.insert(0, '5%')
    four_str.insert(0, '7%')
    six_str.insert(0, '9%')
    eight_str.insert(0, '10%')
    ten_str.insert(0, '20%')

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
    two_str_p.insert(0, '5%')
    four_str_p.insert(0, '7%')
    six_str_p.insert(0, '9%')
    eight_str_p.insert(0, '10%')
    ten_str_p.insert(0, '20%')

    print('Detection Rate')
    print(tabulate([one_str, two_str, four_str, six_str, eight_str, ten_str], headers=headers))
    print("\n \n ")
    print('Precision')
    print(tabulate([one_str_p, two_str_p, four_str_p, six_str_p, eight_str_p, ten_str_p], headers=headers))
else:
    headers = ['FAR', 'PCA30', 'PCA40', 'PCA50', 'PCA60', 'PCA70']

    one_perc = [recall1[i][0] for i in components]
    two_perc = [recall1[i][1] for i in components]
    four_perc = [recall1[i][2] for i in components]
    six_perc = [recall1[i][3] for i in components]
    ten_perc = [recall1[i][4] for i in components]
    thirty_perc = [recall1[i][5] for i in components]

    one_perc_prec = [precision1[i][0] for i in components]
    two_perc_prec = [precision1[i][1] for i in components]
    four_perc_prec = [precision1[i][2] for i in components]
    six_perc_prec = [precision1[i][3] for i in components]
    ten_perc_prec = [precision1[i][4] for i in components]
    thir_perc_prec = [precision1[i][5] for i in components]

    one_perc_prec.insert(0, '1%')
    two_perc_prec.insert(0, '5%')
    four_perc_prec.insert(0, '7%')
    six_perc_prec.insert(0, '9%')
    ten_perc_prec.insert(0, '10%')
    thir_perc_prec.insert(0, '20%')

    one_perc.insert(0, '1%')
    two_perc.insert(0, '5%')
    four_perc.insert(0, '7%')
    six_perc.insert(0, '9%')
    ten_perc.insert(0, '10%')
    thirty_perc.insert(0, '20%')

    ######
    one_perc_f1 = [f1_score1[i][0] for i in components]
    two_perc_f1 = [f1_score1[i][1] for i in components]
    four_perc_f1 = [f1_score1[i][2] for i in components]
    six_perc_f1 = [f1_score1[i][3] for i in components]
    ten_perc_f1 = [f1_score1[i][4] for i in components]
    thir_perc_f1 = [f1_score1[i][5] for i in components]

    one_perc_f1.insert(0, '1%')
    two_perc_f1.insert(0, '5%')
    four_perc_f1.insert(0, '7%')
    six_perc_f1.insert(0, '9%')
    ten_perc_f1.insert(0, '10%')
    thir_perc_f1.insert(0, '20%')

    one_perc_acc = [accuracy1[i][0] for i in components]
    two_perc_acc = [accuracy1[i][1] for i in components]
    four_perc_acc = [accuracy1[i][2] for i in components]
    six_perc_acc = [accuracy1[i][3] for i in components]
    ten_perc_acc = [accuracy1[i][4] for i in components]
    thir_perc_acc = [accuracy1[i][5] for i in components]

    one_perc_acc.insert(0, '1%')
    two_perc_acc.insert(0, '5%')
    four_perc_acc.insert(0, '7%')
    six_perc_acc.insert(0, '9%')
    ten_perc_acc.insert(0, '10%')
    thir_perc_acc.insert(0, '20%')

    print('Detection Rate')
    print(tabulate([one_perc, two_perc, four_perc, six_perc, ten_perc, thirty_perc], headers=headers))
    print("\n \n ")
    print('Precision')
    print(tabulate([one_perc_prec, two_perc_prec, four_perc_prec, six_perc_prec, ten_perc_prec, thir_perc_prec],
                   headers=headers))
    print("\n \n ")
    print('F1_Score')
    print(
        tabulate([one_perc_f1, two_perc_f1, four_perc_f1, six_perc_f1, ten_perc_f1, thir_perc_f1], headers=headers))
    print("\n \n ")
    print('Accuracy')
    print(tabulate([one_perc_acc, two_perc_acc, four_perc_acc, six_perc_acc, ten_perc_acc, thir_perc_acc],
                   headers=headers))

#qpca70.runtime_comparison(1000000, 8000, 'CICIDSExp1.pdf', estimate_components='right_sv', classic_runtime='classic')
#qpca70.runtime_comparison(10000, 100, 'CICIDSExp1_zoom.pdf', estimate_components='right_sv', classic_runtime='classic')
