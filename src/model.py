import numpy as np
import math
from dense import Dense
from dropout import Dropout
from loss import *
from sklearn.datasets import load_breast_cancer


class UniversalModel:
    def __init__(self):
        self.layers = np.array([])
        
    
    def add(self, object):
        if object._type == "Dense":
            if self.layers.shape[0] > 0:
                for layer in self.layers[::-1]:
                    if layer._type == "Dense":
                        _prev_dense_pos = layer
                        break
                object.init_weights(_prev_dense_pos.n_of_neurons)
            self.layers = np.append(self.layers, object)
        elif object._type == "Dropout":
            self.layers = np.append(self.layers, object)
        
    def predict(self, inputs):
        for layer in self.layers:
            if layer._type == "Dropout":
                inputs = layer.dropout(inputs)
            elif layer._type == "Dense":
                inputs = layer.forward(inputs)
        return inputs
    
    def show(self):
        return "UI not implemented yet"
    
    def console(self, text):
        if self.console_state:
            print(text)
        
    
    def train(self, inputs, labels, loss = "mae", learning_rate = 0.0001, batch_size = 30, epochs = 10, console=True):
        self.console_state = console
        """
        Loss Types: mae, mse, accuracy, precision, recall
        """
        _n_samples = inputs.shape[0]
        for epoch in range(epochs):
            index = np.random.permutation(_n_samples)
            _inputs_shuffled = inputs[index]
            _labels_shuffled = labels[index]
            for start in range(0, batch_size):
                end = min(start+batch_size, _n_samples)
                batch_inputs = inputs[start:end]
                batch_labels = labels[start:end]
                
                prediction = self.predict(batch_inputs)
                y = [batch_labels, prediction]
                #implement backprop
                loss_value = get_loss(loss, y)
            self.console(f"Epoch: {epoch+1}/{epochs}; Loss: {loss_value}")
        
        
        
    def backpropagate(self, grad_loss, learning_rate):
        h = 0.00001
        
#TEST

data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1,1)

model = UniversalModel()
model.add(Dense(n_of_neurons=30, bias=True, activation_function="relu", weight_init="xavier"))
model.add(Dense(n_of_neurons=16, bias=True, activation_function="relu", weight_init="xavier"))
model.add(Dropout(dropout_rate=0.2))
model.add(Dense(n_of_neurons=8, bias=True, activation_function="relu", weight_init="xavier"))
model.add(Dense(n_of_neurons=1, bias=True, activation_function="sigmoid", weight_init="xavier"))

model.train(X, y, "mse")
