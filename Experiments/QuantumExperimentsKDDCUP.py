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
#Load data
for i in range(5):
    trains.append(pd.read_csv('../KDDCUP/Trains/trains' + str(i)).drop(columns='Unnamed: 0'))
    tests.append(pd.read_csv('../KDDCUP/Tests/tests' + str(i)).set_index('Unnamed: 0'))

for i in range(5):
    x = tests[i].index
    test.append(df1.loc[x])
    labels.append(df1.loc[x][41])
    trains_without_labels.append(trains[i].drop(columns='41'))

# Compute quantiles
alpha = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
quantils = []
for i in alpha:
    eq = [1, -2, i]
    quantile = 1 - np.round(np.roots(eq)[1], decimals=4)
    quantils.append(quantile)

#Fit PCA models
while True:
    try:
        qpca30 = qPCA(svd_solver='full', name='qpca30').fit(trains[0].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=8, p=0.30,
                                                            theta_minor=np.sqrt(0.20*trains[0].shape[0]),
                                                            estimate_all=True,
                                                            delta=0.9, eps=3, true_tomography=True,
                                                            eta=0.2, spectral_norm_est=False,
                                                            estimate_least_k=False)
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
                                                            delta=0.9, eps=3, true_tomography=True, eta=0.16,
                                                            theta_minor=np.sqrt(0.20*trains[1].shape[0]),
                                                            estimate_least_k=False)
    except:
        pass
    else:
        break
print('PCA40 done', 'n_components', qpca40.topk)

while True:
    try:
        qpca50 = qPCA(svd_solver='full', name='PCA50').fit(trains[2].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.50,
                                                            estimate_all=True,
                                                            delta=0.9, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[2].shape[0]),
                                                            estimate_least_k=False,spectral_norm_est=True)
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
                                                            delta=0.9, eps=50, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[3].shape[0]),
                                                            estimate_least_k=False
                                                            ,spectral_norm_est=True
                                                            )
    except:
        pass
    else:
        break
print('PCA60 done', 'n_components', qpca60.topk)

while True:
    try:
        qpca70 = qPCA(svd_solver='full', name='PCA70').fit(trains[4].drop(columns='41'), theta_estimate=True,
                                                            eps_theta=1,
                                                            p=0.70,
                                                            estimate_all=True,
                                                            delta=0.5, eps=3, true_tomography=True, eta=0.1,
                                                            theta_minor=np.sqrt(0.20 * trains[4].shape[0]),
                                                            spectral_norm_est=True, estimate_least_k=False)
    except:
        pass
    else:
        break
print('PCA70 done', 'n_components', qpca70.topk)

QPCAs = [qpca30, qpca40, qpca50, qpca60, qpca70]
# Make prediction. Set quantum=True for quantum experiment and False for the classical counterparts.
# With experiment=0, we mean the PCA model with only major components. Experiment=1 refers to PCA model with major and
# minor components.
# Only_dot_product=True refers to the experiments where we consider only the dot product in the computations of the model.
# Only_dot_product=False instead refers to the ensemble model.
# name_negative_class-> is the name of the normal label.
qmodel = Model(QPCAs, quantils, quantum=True).fit(trains_without_labels, minor_sv_variance=0.20, only_dot_product=True,
                                                  experiment=0)
recall_dot, precision_dot, accuracy_dot, f1_score_dot = qmodel.predict(tests, labels, name_negative_class='normal.',
                                                                       only_dot_product=True, experiment=0)

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

# Runtime_comparison plot the comparison between classical and quantum model's fitting. The first two parameters are the
# maximum number of samples and features respectively.
# estimate_components='right_sv' means that in the top-k singular vectors extraction, we extract only the right singular
# vectors.
# classic_runtime='rand'-> randomized classical running time. Otherwise, 'classic' refers to the full SVD complexity.

#qpca50.runtime_comparison(5000000, 500, 'KDDUCUP01.pdf', estimate_components='right_sv', classic_runtime='rand')
qpca70.runtime_comparison(10000000, 50, 'KDDUCUP1.pdf', estimate_components='right_sv', classic_runtime='rand')

