import numpy as np
from object import *

def get_activation(_type):
    activations = {
        "relu": lambda x: np.maximum(0, x),
        "sigmoid": lambda x: 1/(1+np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "linear": lambda x: x
    }
    return activations.get(_type.lower(), lambda x: x)

def get_weight_init(_type, _shape):
    weight_inits = {
        "zero": lambda shape: np.zeros(shape),
        "xavier": lambda shape: np.random.randn(*shape) * np.sqrt(1. / shape[1]),
        "he": lambda shape: np.random.randn(*shape) * np.sqrt(2. / shape[1])
    }
    return weight_inits.get(_type.lower(), lambda shape: np.random.randn(*shape))(_shape)


class Dense(Object):
    def __init__(self, n_of_neurons,  bias=False, activation_function="ReLU", weight_init="zero"):
        super().__init__("Dense")
        self.n_of_neurons = n_of_neurons
        self.activation_type = activation_function
        self.activation = get_activation(self.activation_type)
        self.weight_init = weight_init
        self.bias = np.random.randn(self.n_of_neurons, 1) if bias else np.zeros((self.n_of_neurons, 1))
        self.weights = np.array([])

    def init_weights(self, prev_n_of_neurons):
        self.weights = get_weight_init(self.weight_init, (self.n_of_neurons, prev_n_of_neurons))
        
    def __repr__(self):
        return f"Dense(n_of_neurons={self.n_of_neurons},bias={True if self.bias.any() else False},activation={self.activation_type})"
    
    def forward(self, inputs):
        if self.weights.shape[0] == 0:
            return inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation(z) if self.activation is not None else z
        return self.output



