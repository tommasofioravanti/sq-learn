import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, qPCA
from sklearn.QuantumUtility.Utility import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler, QuantileTransformer
import warnings

warnings.filterwarnings("ignore")
# df_new -> LEN TRAIN 50000, LEN_VALIDATION 226966, thr=0.4259394035517815, n_q=751,n_c=12
# df_Ddos -> LEN_TRAIN 50000, LEN_VALIDATION 90000
# df_DdoSPaper -> LEN_TRAIN 158022, LEN_VALIDATION 60000, th=0.06632108379654125, n_q=24,n_c=32
df_new = pd.read_csv('../df_new.csv')

df_new.replace('Infinity', -1, inplace=True)
df_new[["Flow Bytes/s", "Flow Packets/s"]] = df_new[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)
df_new.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

string_features = list(df_new.select_dtypes(include=['object']).columns)
string_features.remove('Label')
le = LabelEncoder()
df_new[string_features] = df_new[string_features].apply(lambda col: le.fit_transform(col))

df_copy = df_new.drop(columns='Label')
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(df_copy)
constant_columns = [column for column in df_copy.columns if
                    column not in df_copy.columns[constant_filter.get_support()]]
categorical_columns_train = ['Flow ID', 'Source IP', 'Source Port',
                             'Destination IP', 'Destination Port', 'Timestamp', 'Label']

categorical_columns_test = ['Flow ID', 'Source IP', 'Source Port',
                            'Destination IP', 'Destination Port', 'Timestamp']

columns_to_drop_train = categorical_columns_train + constant_columns

columns_to_drop_test = categorical_columns_test + constant_columns
df_new = df_new.drop(columns=columns_to_drop_test)

LEN_TRAIN = 50000
LEN_VALIDATION = 226966

#Hyperparamters found with Optuna

n_quantils = 751
threshold = 0.4259394035517815
n_components = 12
qt = QuantileTransformer(n_quantiles=n_quantils, random_state=0)
x = qt.fit_transform(df_new.drop(columns='Label'))

train = x[:LEN_TRAIN]
test = x[LEN_TRAIN + LEN_VALIDATION:]

# Score on Test set
qpca = qPCA(svd_solver="full", name='PCA')
delta = [0.01, 0.1, 0.9, 2]
for d_ in delta:
    print(d_)

    n_c = 100
    while True:
        try:

            qPca_fitted = qpca.fit(train, theta_estimate=False, eps_theta=3, p=n_components,
                                   estimate_all=True, delta=0, eps=0, true_tomography=True,
                                   eta=0.04, norm='L2', condition_number_est=False, spectral_norm_est=False)
            n_c = qPca_fitted.topk
            print(n_c)
            if n_c <= 12:
                break
        except:
            pass

    print('quantum_components_retained:', qPca_fitted.topk)
    print('classic_components_retained:', qPca_fitted.components_retained_)
    print('norm for muA:', qPca_fitted.norm_muA)
    transform_X = qPca_fitted.transform(test, classic_transform=False, use_classical_components=False)

    inverse_X = qPca_fitted.inverse_transform(transform_X, use_classical_components=False)
    loss = np.sum((test - inverse_X) ** 2, axis=1)

    attack_prediction = df_new[LEN_TRAIN + LEN_VALIDATION:].iloc[np.where(loss > threshold)[0]]['Label'].value_counts()
    normal_prediction = df_new[LEN_TRAIN + LEN_VALIDATION:].iloc[np.where(loss <= threshold)[0]]['Label'].value_counts()

    total_predicted_attack = attack_prediction.sum()
    if len(attack_prediction) > 0:
        FP = attack_prediction['BENIGN']
        TP = total_predicted_attack - FP
    else:
        FP = 0
        TP = 0

    total_predicted_negative = normal_prediction.sum()
    if len(normal_prediction) > 0:
        TN = normal_prediction['BENIGN']
        FN = total_predicted_negative - TN
    else:
        TN = 0
        FN = 0

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('f1:', 2 / ((1 / p) + (1 / r)), 'precision:', p, 'recall:', r, 'accuracy:', accuracy)
    # if d_ == 0.1:
    #    qPca_fitted.runtime_comparison(1000000000, 5000, 'CICIDSLoss0_1Zoom.pdf', estimate_components='right_sv',
    #                                   classic_runtime='rand')
