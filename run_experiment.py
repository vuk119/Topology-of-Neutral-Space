import numpy as np
import matplotlib.pyplot as plt

from binary_inputs import b_input
from perceptron import perceptron


'''
Parameters

n - input size
bias - True if to include else False
sample_size - number of points along each axis
parameter_limit - sample interval is from -parameter_limit up to +parameter_limit

'''
n = 3
bias = False
sample_size = 10
parameter_limit = 1

'''
Initialize perceptron and input
'''
p = perceptron(n)
b_in = b_input(n)


'''
Initialize parameter - class matrix
'''
parameter_space = np.ndarray((sample_size, sample_size, sample_size))

for i in range(sample_size):
    for j in range(sample_size):
        for k in range(sample_size):
            w1 = 2*i*parameter_limit/sample_size - parameter_limit
            w2 = 2*j*parameter_limit/sample_size - parameter_limit
            w3 = 2*k*parameter_limit/sample_size - parameter_limit
            p.set_weights([w1,w2, w3], 0)
            prediction = p.predict(b_in.inputs)
            parameter_space[i,j, k] = b_in.output_to_class[prediction]


plt.imshow(parameter_space[2,:,:])
plt.show()
