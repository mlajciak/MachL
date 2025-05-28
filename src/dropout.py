import numpy as np
import math
import random
from object import *

class Dropout(Object):
    def __init__(self, dropout_rate):
        super().__init__("Dropout")
        self.dropout_rate = dropout_rate
        
    def dropout(self, inputs):
        _n_of_dropped = int(inputs.shape[0]*self.dropout_rate)
        _dropped = [random.randint(0,inputs.shape[0]-1) for d in range(_n_of_dropped)]
        
        for drop in _dropped:
            inputs[drop] = 0
            
        return inputs