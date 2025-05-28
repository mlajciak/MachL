import numpy as np



def get_loss(_type, y):
    
    losses = {
        "mae":lambda y: MAE(y),
        "mse":lambda y: MSE(y),
        "accuracy":lambda y: accuracy(y),
        "precision":lambda y: precision(y),
        "recall":lambda y: recall(y),
        }
    
    return losses.get(_type.lower(), lambda y: y)

def MSE(self, y):
    return (y[0] - y[1])**2

def MAE(self, y, n_of_data):
    return (1/n_of_data) * (sum(y[1] - y[0]))

def accuracy(self, confusion_matrix):
    """
    confusion_matrix - 2x2, assuming that np array
    
    TP FP
    
    FN TN
    
    
    accuracy = (TP+TN)/(TP+TN+FP+FN) = (correct predictions)/(all_predictions)
    """
    return (confusion_matrix[0][0]+confusion_matrix[1][1])/(np.sum(confusion_matrix))

def precision (self, confusion_matrix):
    
    """
    precision = TP/(TP+FP)
    """
    
    return (confusion_matrix[0][0])/(confusion_matrix[0][0] + confusion_matrix[0][1])

def recall(self, confusion_matrix):
    
    """
    recall = TP/(TP+FN) 
    """
    return (confusion_matrix[0][0])/(confusion_matrix[0][0] + confusion_matrix[1][0])

    