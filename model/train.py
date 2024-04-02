import numpy as np

class perceptron:
    def __init__(self, nb, init_weight, activation):
        self.id = nb
        self.weight = init_weight
        self.activation = activation
    def hello(self):
        print("HEllo")

class layers:
    def __init__(self):
        self.layers = []

    def DenseLayer(self, nb_perceptron, activation, weights_initializer):
        perceptrons = []
        for i in range(nb_perceptron):
            perceptrons.append(perceptron(i, self.init_weight(), activation))
        self.layers.append(perceptrons)
        print(self.layers)
        
    def init_weight(self):
        return 0

    def sigmoid(self, x):
        return (1 / 1 + np.exp(-x))


class model:
    def __init__():
        print("gello")
