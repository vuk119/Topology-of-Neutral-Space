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
n = 4
bias = False
sample_size = 10
parameter_limit = 1

'''
Initialize perceptron and input
'''
p = perceptron(n)
b_in = b_input(n)

#print(b_in.output_to_class)



'''
Initialize parameter - class matrix
'''
parameter_space = np.ndarray((sample_size, sample_size, sample_size, sample_size), dtype=int)

print('initialized')

for i in range(sample_size):
    for j in range(sample_size):
        for k in range(sample_size):
            for m in range(sample_size):
                w1 = 2*i*parameter_limit/sample_size - parameter_limit
                w2 = 2*j*parameter_limit/sample_size - parameter_limit
                w3 = 2*k*parameter_limit/sample_size - parameter_limit
                w4 = 2*m*parameter_limit/sample_size - parameter_limit
                p.set_weights([w1,w2,w3, w4], 0)
                prediction = p.predict(b_in.inputs)
                parameter_space[i,j,k,m] = b_in.output_to_class[prediction]

print('assigned')

class_counts = [np.sum(np.sum(np.sum(np.sum(np.sum(parameter_space==i))))) for i in range(2**2**n)]

print('counted')

print(np.sum(class_counts))
plt.scatter(range(2**2**n),np.round(np.array(class_counts)/sample_size**n,2))
plt.show()

print(class_counts)

assert False

'''
Find the ratio of changes without going outside the class and the number of changes
'''

def check(i, sample_size):
    return (i>=0 and i<sample_size)

local_robustness = np.zeros((sample_size, sample_size))
class_robustness = [0 for i in range(2**2**n)]


for i in range(sample_size):
    for j in range(sample_size):
        neighbours_cnt = 0

        '''
        Check all the neighbours
        '''
        for di in range(-1,2):
            for dj in range(-1, 2):
                if di!=0 or dj!=0:
                    if check(i+di, sample_size) and check(j+dj, sample_size):
                        neighbours_cnt += 1
                        if parameter_space[i+di, j+dj] == parameter_space[i,j]:
                            local_robustness[i,j]+=1

        local_robustness[i,j]/=neighbours_cnt
        class_robustness[parameter_space[i,j]]+=local_robustness

print(parameter_space)
plt.imshow(np.rot90(parameter_space,3))
plt.show()
