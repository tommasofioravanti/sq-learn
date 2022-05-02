import numpy as np
from sklearn.QuantumUtility.Utility import *

sv = [1, 0.978, 0.96, 0.85, 0.6, 0.03]
epsilon = 0.1
# median = median_evaluation(amplitude_estimation, (0.978, epsilon), gamma=0.1)
thetas = [wrapper_phase_est_arguments(sv_) / np.pi for sv_ in sv]
#thetas = [1]
# Set M to even value to estimate theta=1 with certainity.
estimations = [amplitude_estimation(theta, M=158, epsilon=epsilon) for theta in thetas]

print(estimations)
