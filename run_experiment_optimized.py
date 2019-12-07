import numpy as np
import matplotlib.pyplot as plt
import time

from binary_inputs import b_input
from perceptron import perceptron

start_time = time.time()

'''
Parameters

n - input size
bias - True if to include else False
sample_size - number of points along each axis
parameter_limit - sample interval is from -parameter_limit up to +parameter_limit

'''
n =5
bias = False
sample_size = 40
parameter_limit = 1

'''
Initialize perceptron and input
'''
p = perceptron(n)
b_in = b_input(n)


'''
Initialize parameter - class matrix
'''
parameter_space = np.ndarray([sample_size]*n, dtype=int)

neutral_set_dict = {}
neutral_set_list = []

print('initialized', time.time()-start_time)

for number in range(sample_size**n):
    indices = []

    for _ in range(n):
        indices.append(number%sample_size)
        number = number // sample_size

    parameters = [2*i*parameter_limit/sample_size - parameter_limit for i in indices]
    p.set_weights(parameters, 0)
    prediction = p.predict(b_in.inputs)

    if prediction not in neutral_set_list:
            neutral_set_list.append(tuple(prediction))
            neutral_set_dict[tuple(prediction)] = len(neutral_set_list)-1

    parameter_space[tuple(indices)] = neutral_set_dict[tuple(prediction)]

print('assigned', time.time()-start_time)

neutral_set_counts = [np.sum(np.sum(np.sum(np.sum(np.sum(parameter_space==i))))) for i in range(len(neutral_set_list))]

print('counted', time.time()-start_time)




'''
Find the ratio of changes without going outside the class and the number of changes
'''

def check(i, sample_size):
    return (i>=0 and i<sample_size)

local_robustness = np.zeros([sample_size]*n)
neutral_set_robustness = [0 for i in range(len(neutral_set_list))]


for number in range(sample_size**n):
    indices = []

    for _ in range(n):
        indices.append(number%sample_size)
        number = number // sample_size

    neighbours_cnt = 0

    '''
    Check all the neighbours
    '''

    '''
    See all neighbours
    '''
    for number2 in range(3**n):

        #Determine the shifts
        indices_shifts  = []
        valid_shifts = True
        for _ in range(n):
            shift = number2%3-1

            if indices[_]+shift<0 or indices[_]+shift>=sample_size:
                valid_shifts = False
                break

            indices_shifts.append(shift)
            number2 = number2 // sample_size

        if not valid_shifts:
            continue

        neighbours_cnt+=1

        #Increase the robostness
        if parameter_space[tuple([indices[i]+indices_shifts[i] for i in range(len(indices))])] == parameter_space[tuple(indices)]:
            local_robustness[tuple(indices)]+=1

    #Decrease by itself
    local_robustness[tuple(indices)]-=1
    neighbours_cnt-=1
    local_robustness[tuple(indices)] /= neighbours_cnt
    neutral_set_robustness[parameter_space[tuple(indices)]] += local_robustness[tuple(indices)]

neutral_set_robustness = [neutral_set_robustness[i] / neutral_set_counts[i] if neutral_set_counts[i]!=0 else 0 for i in range(len(neutral_set_counts))]

#print(neutral_set_counts)
#print(neutral_set_robustness)


neutral_set_volumes = [neutral_set_counts[i] / sum(neutral_set_counts) for i in range(len(neutral_set_counts))]

print('Done', time.time()-start_time)

plot_x = neutral_set_volumes
plot_y = neutral_set_robustness

with open("dim{}sample{}.txt".format(n,sample_size), 'w') as file:
    for i in range(len(neutral_set_volumes)):
        file.write(str(neutral_set_volumes[i]))
        file.write(',')
        file.write(str( neutral_set_robustness[i]))
        file.write('\n')

plt.scatter(plot_x, plot_y)
plt.xlim([0,2 / len(neutral_set_list)])
plt.ylim([0,1])
plt.show()
