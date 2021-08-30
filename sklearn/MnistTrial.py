from sklearn.decomposition import qPCA

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
pca = qPCA(svd_solver="full")
pca.n_components = 61
#k_sqrt = np.sqrt(pca.n_components)

pca_model = pca.fit(X)
error=0.8
#Transform the features
X_train_pca = pca_model.transform(X,classic_transform=False, epsilon_delta=error,quantum_representation=True, norm='est_representation',
                                  tomography=True)

knn = KNeighborsClassifier(n_neighbors=7)
score = cross_validate(knn, X_train_pca['quantum_representation_results'][0],y,cv=StratifiedKFold(n_splits=10,shuffle=True, random_state=1234), n_jobs=4)
accuracies=[]
accuracies.append([error, X_train_pca['quantum_representation_results'][2], np.average(score['test_score'])])

print(f"10-fold Cross-validation - Estimated UE")
print(f"(delta + epsilon): {error}")
print(f"Error-F_norm-Accuracy: {accuracies}")
