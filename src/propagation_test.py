from sklearn.datasets import load_breast_cancer
from .model import UniversalModel
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

