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
trains_without_labels = []
tests = []
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

while True:
    try:
        qpca30 = qPCA(svd_solver='full', name='qpca30').fit(trains[0].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=8, p=0.30,
                                                            theta_minor=np.sqrt(0.20*trains[0].shape[0]),
                                                            estimate_all=True,
                                                            delta=0.9, eps=3, true_tomography=True,
                                                            eta=0.2,
                                                            estimate_least_k=True)
    except:
        pass
    else:
        break
print('PCA30 done', 'n_components', qpca30.topk)

while True:
    try:
        qpca40 = qPCA(svd_solver='full', name='qpca40').fit(trains[1].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.40,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20*trains[1].shape[0]),
                                                            estimate_least_k=True)
    except:
        pass
    else:
        break
print('PCA40 done', 'n_components', qpca40.topk)
while True:
    try:
        qpca50 = qPCA(svd_solver='full', name='qpca50').fit(trains[2].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.50,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[2].shape[0]),
                                                            estimate_least_k=True
                                                            )
    except:
        pass
    else:
        break
print('PCA50 done', 'n_components', qpca50.topk)
while True:
    try:
        qpca60 = qPCA(svd_solver='full', name='qpca60').fit(trains[3].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.60,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[3].shape[0]),
                                                            estimate_least_k=True
                                                            )
    except:
        pass
    else:
        break
print('PCA60 done', 'n_components', qpca60.topk)
while True:
    try:
        qpca70 = qPCA(svd_solver='full', name='qpca70').fit(trains[4].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.70,
                                                            estimate_all=True,
                                                            delta=0.1, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[4].shape[0]),
                                                            estimate_least_k=True
                                                            )
    except:
        pass
    else:
        break
print('PCA70 done', 'n_components', qpca70.topk)

QPCAs = [qpca30, qpca40, qpca50, qpca60, qpca70]
#QPCAs = [qpca30]

qmodel = Model(QPCAs, quantils, quantum=True).fit(trains_without_labels, minor_sv_variance=0.20, only_dot_product=True,
                                                  experiment=1)
recall_dot, precision_dot, accuracy_dot, f1_score_dot = qmodel.predict(tests, labels, name_negative_class='normal.',
                                                                       only_dot_product=True, experiment=1)

components = [i for i in recall_dot]
headers = ['FAR', 'PCA30', 'PCA40', 'PCA50', 'PCA60', 'PCA70']

one_perc = [recall_dot[i][0] for i in components]
two_perc = [recall_dot[i][1] for i in components]
four_perc = [recall_dot[i][2] for i in components]
six_perc = [recall_dot[i][3] for i in components]
ten_perc = [recall_dot[i][4] for i in components]
thirty_perc = [recall_dot[i][5] for i in components]

one_perc_prec = [precision_dot[i][0] for i in components]
two_perc_prec = [precision_dot[i][1] for i in components]
four_perc_prec = [precision_dot[i][2] for i in components]
six_perc_prec = [precision_dot[i][3] for i in components]
ten_perc_prec = [precision_dot[i][4] for i in components]
thir_perc_prec = [precision_dot[i][5] for i in components]

one_perc_prec.insert(0, '1%')
two_perc_prec.insert(0, '2%')
four_perc_prec.insert(0, '4%')
six_perc_prec.insert(0, '6%')
ten_perc_prec.insert(0, '8%')
thir_perc_prec.insert(0, '10%')

one_perc.insert(0, '1%')
two_perc.insert(0, '2%')
four_perc.insert(0, '4%')
six_perc.insert(0, '6%')
ten_perc.insert(0, '8%')
thirty_perc.insert(0, '10%')

######
one_perc_f1 = [f1_score_dot[i][0] for i in components]
two_perc_f1 = [f1_score_dot[i][1] for i in components]
four_perc_f1 = [f1_score_dot[i][2] for i in components]
six_perc_f1 = [f1_score_dot[i][3] for i in components]
ten_perc_f1 = [f1_score_dot[i][4] for i in components]
thir_perc_f1 = [f1_score_dot[i][5] for i in components]

one_perc_f1.insert(0, '1%')
two_perc_f1.insert(0, '2%')
four_perc_f1.insert(0, '4%')
six_perc_f1.insert(0, '6%')
ten_perc_f1.insert(0, '8%')
thir_perc_f1.insert(0, '10%')

one_perc_acc = [accuracy_dot[i][0] for i in components]
two_perc_acc = [accuracy_dot[i][1] for i in components]
four_perc_acc = [accuracy_dot[i][2] for i in components]
six_perc_acc = [accuracy_dot[i][3] for i in components]
ten_perc_acc = [accuracy_dot[i][4] for i in components]
thir_perc_acc = [accuracy_dot[i][5] for i in components]

one_perc_acc.insert(0, '1%')
two_perc_acc.insert(0, '2%')
four_perc_acc.insert(0, '4%')
six_perc_acc.insert(0, '6%')
ten_perc_acc.insert(0, '8%')
thir_perc_acc.insert(0, '10%')

print('Detection Rate')
print(tabulate([one_perc, two_perc, four_perc, six_perc, ten_perc, thirty_perc], headers=headers))
print("\n \n ")
print('Precision')
print(tabulate([one_perc_prec, two_perc_prec, four_perc_prec, six_perc_prec, ten_perc_prec, thir_perc_prec],
               headers=headers))
print("\n \n ")
print('F1_Score')
print(tabulate([one_perc_f1, two_perc_f1, four_perc_f1, six_perc_f1, ten_perc_f1, thir_perc_f1], headers=headers))
print("\n \n ")
print('Accuracy')
print(tabulate([one_perc_acc, two_perc_acc, four_perc_acc, six_perc_acc, ten_perc_acc, thir_perc_acc], headers=headers))
# qpca70.runtime_comparison(10000, 10, 'KDDUCUP0.pdf', estimate_components='right_sv', classic_runtime='rand')
'''dictionary_major = {}
dictionary_minor = {}
for e, pca in enumerate(QPCAs):

    out_threshold_list_major = []
    out_threshold_list_minor = []
    emp_distribution_major = []

    for j in range(len(trains[e].drop(columns='41'))):
        sample = np.array(trains[e].drop(columns='41').iloc[j])
        y = np.dot(sample, pca.estimate_right_sv.T)
        s_major = np.sum(y ** 2 / pca.estimate_fs)

        emp_distribution_major.append(s_major)

    for q in quantils:
        n_major = len(emp_distribution_major)

        sort_major = sorted(emp_distribution_major)

        out_threshold_major = sort_major[int(n_major * q)]

        out_threshold_list_major.append(out_threshold_major)

    dictionary_major.update({pca: out_threshold_list_major})
    # dictionary_minor.update({pca.components_retained_:out_threshold_list_minor})

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
# First experiment
components = [i for i in classification_res]


headers = ['FAR','PCA30','PCA40','PCA50','PCA60','PCA70']
one_perc = [classification_res[i][0] for i in components]
two_perc = [classification_res[i][1] for i in components]
four_perc = [classification_res[i][2] for i in components]
six_perc = [classification_res[i][3] for i in components]
eight_perc = [classification_res[i][4] for i in components]
ten_perc = [classification_res[i][5] for i in components]


one_perc.insert(0,'1%')
two_perc.insert(0,'2%')
four_perc.insert(0,'4%')
six_perc.insert(0,'6%')
eight_perc.insert(0,'8%')
ten_perc.insert(0,'10%')

print(tabulate([one_perc, two_perc, four_perc, six_perc, eight_perc, ten_perc], headers=headers))'''
