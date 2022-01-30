import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA, qPCA

from sklearn.QuantumUtility.Utility import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler, QuantileTransformer
import warnings

warnings.filterwarnings("ignore")
# df_Darknet -> LEN TRAIN 50000, LEN_VALIDATION 30000 threshld 0.4438816547139376, n_c=35
#df_darknet_unbalanced->train 50000, validation 15000 threshold 0.39397872912544213, n_c=27, n_q=446
#
df1 = pd.read_csv('../df_Darknet_unbalanced.csv')

df1.columns = df1.columns.str.strip()
df1 = df1.drop(df1[pd.isnull(df1['Flow ID'])].index)
df1.replace('Infinity', -1, inplace=True)
df1[["Flow Bytes/s", "Flow Packets/s"]] = df1[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)
df1.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
string_features = list(df1.select_dtypes(include=['object']).columns)
# string_features.remove('Label')
le = LabelEncoder()
df1[string_features] = df1[string_features].apply(lambda col: le.fit_transform(col))
# REMOVE CONSTANT FEATURES
df_copy = df1.drop(columns='Label')
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(df_copy)
constant_columns = [column for column in df_copy.columns if
                    column not in df_copy.columns[constant_filter.get_support()]]

categorical_columns_train = ['Flow ID', 'Src IP', 'Src Port',
                             'Dst IP', 'Dst Port', 'Label.1', 'Timestamp']

columns_to_drop_train = categorical_columns_train + constant_columns

df_new = df1.drop(columns=columns_to_drop_train)

LEN_TRAIN = 50000
LEN_VALIDATION = 15000

t = 0.39397872912544213
n_components = 27

qt = StandardScaler()
x = qt.fit_transform(df_new.drop(columns='Label'))

train = x[:LEN_TRAIN]
test = x[LEN_TRAIN + LEN_VALIDATION:]

pca = qPCA(svd_solver="full")
n_c = 100
delta = [0.1,0.9, 2]
for d_ in delta:

    while True:
        try:

            qPca_fitted = pca.fit(train, theta_estimate=True, eps_theta=0.10, p=n_components,
                                  estimate_all=True, delta=d_, eps=0.1,
                                  eta=0.004)
            n_c = qPca_fitted.topk
            print(n_c)
        except:
            pass
        else:
            if n_c > 27:
                pass
            else:
                break



    print('quantum_components_retained:', qPca_fitted.topk)
    print('classic_components_retained:', qPca_fitted.components_retained_)
    transform_X = qPca_fitted.transform(test, classic_transform=False, use_classical_components=False)

    inverse_X = qPca_fitted.inverse_transform(transform_X, use_classical_components=False)
    loss = np.sum((test - inverse_X) ** 2, axis=1)

    attack_prediction = df_new[LEN_TRAIN + LEN_VALIDATION:].iloc[np.where(loss > t)[0]]['Label'].value_counts()
    normal_prediction = df_new[LEN_TRAIN + LEN_VALIDATION:].iloc[np.where(loss <= t)[0]]['Label'].value_counts()

    total_predicted_attack = attack_prediction.sum()
    if len(attack_prediction) > 0:
        FP = attack_prediction[0]
        TP = total_predicted_attack - FP
    else:
        FP = 0
        TP = 0

    total_predicted_negative = normal_prediction.sum()
    if len(normal_prediction) > 0:
        TN = normal_prediction[0]
        FN = total_predicted_negative - TN
    else:
        TN = 0
        FN = 0

    # Delta = 0.01, eps_theta = 0.0005
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('f1:', 2 / ((1 / p) + (1 / r)), 'precision:', p, 'recall:', r, 'accuracy:', accuracy)
    if d_==0.1:
        qPca_fitted.runtime_comparison(1000000, 8000, 'DARKNETLoss.pdf', estimate_components='right_sv', classic_runtime='rand')
