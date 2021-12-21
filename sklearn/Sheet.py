from QuantumUtility.Utility import *

import numpy as np
import os
from matplotlib import pyplot as plt
cwd =  os.getcwd()
#largest_sv_path = os.path.join(cwd, 'decomposition/larg_sv1.npy')
#largest_sv = np.load(file = largest_sv_path)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

qpca = qPCA(svd_solver='full')
qpca.fit(X,eps=0.1,theta_estimate=True,eps_theta=0.05,p=0.70,estimate_all=True,delta = 0.5)