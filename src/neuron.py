import numpy as np


def ReLU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Neuron:
    def __init__(self, weights = np.array([]), bias = 0, activation_function=sigmoid):
        self.weights = weights
        self.bias = bias
        self.activation = activation_function
        
    def __repr__(self):
        return f"Neuron(bias={self.bias}, weights={self.weights})"
    
    def forward(self, inputs):
        dot = np.dot(self.weights, inputs) + self.bias
        return self.activation(dot)


