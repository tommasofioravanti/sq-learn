from QuantumUtility.Utility import *

import numpy as np
import os
from matplotlib import pyplot as plt
cwd =  os.getcwd()
#largest_sv_path = os.path.join(cwd, 'decomposition/larg_sv1.npy')
#largest_sv = np.load(file = largest_sv_path)
v=np.load('sparse_arr.npy')
v1 = np.load('array784.npy')
print(np.linalg.norm(v, ord=2))

#print(np.any(np.absolute(v) < 1e-2))


#print(len(v[np.where(np.absolute(v)<1e-1)]))
dict_res = L2_tomogrphy_parallel(v1, delta=0.1, n_jobs=-1)
measurements = list(dict_res.keys())
samples = list(dict_res.values())


#print(samples[0])
for i in range(len(samples)):
    samples[i] = np.linalg.norm(v1 - samples[i],ord=2)

#plt.hist(samples, bins=50, color="darkblue")

plt.plot(measurements,samples)

plt.show()

