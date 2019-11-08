import numpy as np



class perceptron:

    def __init__(self, input_size):

        self.n = input_size

    def set_weights(self, w, b):

        assert len(w)==self.n, "Weights should be a list of length {}".format(self.n+1)

        self.w = w
        self.b = b


    def predict(self, inputs):
        '''
        Expects list of tuples or lists.
        Returns a tuple of outputs
        '''

        outputs = []
        for x in inputs:
            outputs.append(self.hard_sigmoid(np.dot(x, self.w)+self.b))

        return tuple(outputs)

    def soft_sigmoid(self, x):

        return 1/(1+np.exp(-x))

    def hard_sigmoid(self, x):

        return 1 if 1/(1+np.exp(-x))>=0.5 else 0
