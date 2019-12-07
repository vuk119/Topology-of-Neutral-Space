import numpy as np
import itertools



class b_input:

    def __init__(self, size):
        '''
        Object that holds all possible binary inputs of size n and maps them.

        size - size of the input
        '''

        self.n = size
        self.inputs = self.generate_inputs()

        self.inputs.sort(key = lambda x: sum(x))
        #self.class_to_output, self.output_to_class = self.generate_output_class_maps()

    def generate_inputs(self):
        '''
        Generates all the possible inputs of size n.
        '''

        return list(itertools.product([0, 1], repeat=self.n))

    def generate_output_class_maps(self):
        '''
        Generates output class and class output maps.

        There are 2^2^n possible output classes.
        '''
        output_class_map = {}
        class_output_map = {}

        outputs = list(itertools.product([0, 1], repeat=2**self.n))
        classes = list(range(0,2**(2**self.n)))

        output_to_class = dict(zip(outputs,classes))
        class_to_output = dict(zip(classes, outputs))

        return class_to_output, output_to_class
    def get_output_from_class(self, clas):
        return self.class_to_output[clas]
    def get_class_from_output(self, output):
        return self.output_to_class[output]
