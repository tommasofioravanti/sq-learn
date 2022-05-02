from sklearn.QuantumUtility.Utility import *
from sklearn.preprocessing import QuantileTransformer
import pandas as pd

epsilon = 0.05
x = np.array([5.88414114, 2.0327562, 1.68155901, 7.91848042, 1.61922687])
y = np.array([5.15610287, 7.2034771, 9.88496245, 3.46281654, 4.20607662])
# Set the number of iteration Q for median evaluation
est = ipe(x, y, epsilon, Q=5)
print(np.inner(x, y), est)
