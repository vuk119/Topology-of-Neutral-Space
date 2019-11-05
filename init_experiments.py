import numpy as np
import matplotlib.pyplot as plt



"""

Map: sigmoid(a*x+b)
Input space: -1, 0, 1

Different functions:
1. 000      5. 111
2. 001      6. 110
3. 010      7. 101
4. 100      8. 011

Task: Vary 'a' and 'b' and classify into one of 8 classes
Visualizations: 2D plot in 'a-b' space

"""


inputs = [-1, 0, 1]
classes = {(0,0,0):1, (0,0,1):2, (0,1,0):3, (1,0,0):4,
           (1,1,1):5, (1,1,0):6, (1,0,1):7, (0,1,1):8,}

def classify(a, b):
    global inputs
    global classes

    results = tuple([0 if a*X+b<0 else 1 for X in inputs ])

    return classes[results]


size = 1
n = 30

parameters = [2*i/n*size - size for i in range(n+1)]

parametersA = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
parametersB = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

for a in parameters:
    for b in parameters:
        resulting_class = classify(a,b)
        parametersA[resulting_class].append(a)
        parametersB[resulting_class].append(b)


for i in range(1,9):
    plt.scatter( parametersB[i], parametersA[i],label = i)

plt.legend(title='Classes')
plt.show()
