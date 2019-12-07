












from binary_inputs import b_input
from perceptron import perceptron



n =3
bias = False
sample_size = 250
parameter_limit = 1

'''
Initialize perceptron and input
'''
p = perceptron(n)
b_in = b_input(n)



s = set()

positive = [1,3,5,8,10]
negative = [-1,-3,-5,-8,-10]

for i in positive:
    for j in negative:
        for k in negative:

                p.set_weights([i,j,k], 0)
                prediction = p.predict(b_in.inputs)
                s.add(prediction)



print(len(s))
for a in s:
    print(a[1:])
