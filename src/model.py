import numpy as np
import math
from dense import *




class UniversalModel:
    def __init__(self):
        self.layers = np.array([])
    
    def addL(self, layer):
        if self.layers.shape[0] > 0:
            layer.init_weights(self.layers[-1].n_of_neurons)
        self.layers = np.append(self.layers, layer)
        

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def show(self):
        return "UI not implemented yet"
        
    def train(self):
        pass
    
    
model = UniversalModel()
model.addL(Dense(n_of_neurons=32, bias=True, activation_function="relu", weight_init="xavier"))
model.addL(Dense(n_of_neurons=16, bias=True, activation_function="relu", weight_init="xavier"))
model.addL(Dense(n_of_neurons=8, bias=True, activation_function="relu", weight_init="xavier"))
model.addL(Dense(n_of_neurons=1, bias=True, activation_function="sigmoid", weight_init="xavier"))

print(model.predict(np.arange(1, 33).reshape(-1, 1)))






