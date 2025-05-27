import numpy as np
import math
from dense import *




class UniversalModel:
    def __init__(self):
        self.layers = np.array([])
    
    def addL(self, layer):
        self.layers = np.append(self.layers, layer)

    def predict(self, inputs):
        for i,layer in enumerate(self.layers):
            inputs = layer.forward(inputs) if i > 1 else inputs
            print(inputs.shape)
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

print(model.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])))





