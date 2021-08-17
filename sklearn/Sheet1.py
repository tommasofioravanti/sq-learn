from QuantumUtility.Utility import *
import time



import numpy as np
import os
from matplotlib import pyplot as plt
cwd =  os.getcwd()

#v = create_rand_vec(1, 784)
#np.save('sparse_arr.npy',v[0])

#print(samples[0])


v=np.load('sparse_arr.npy')

print(np.linalg.norm(v, ord=2))
samples = []
for i in range(1000):
    # Create a noisy copy
    start = time.time()


    #B = L2_tomographyVector_rightSign(v, delta=0.1)
    B = make_noisy_vec(v,noise=0.1,unitary=True)
    end = time.time()
    #print(i, end - start)
    # Append the Frobenius norm of A-B to the samples
    samples.append(np.linalg.norm(v - B,ord=2))

# Plot the samples
plt.hist(samples, bins=50, color="darkblue")
plt.xlabel(r"$||\mathbf{v} - \overline{\mathbf{v}}||_F$")
plt.ylabel("measurements")
plt.savefig('error_vector.pdf', bbox_inches='tight')