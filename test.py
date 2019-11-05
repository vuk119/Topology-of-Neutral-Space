from binary_inputs import b_input
from perceptron import perceptron



p = perceptron(2)
p.set_weights([2,3], 1)

a = b_input(2)
print(a.inputs)

print(p.predict(a.inputs))

for i in range(20):
    print(2**(2**i))
