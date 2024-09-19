import numpy as np
from nn.types import LossFunction


class CrossEntropy(LossFunction):
    
    def __init__(self):
        pass
    
    def loss(self, probs, labels):
        N = labels.shape[0]
        loss = -np.sum(labels * np.log(probs + 1e-8)) / N
        return loss
    
    def grad(self, probs, labels):
        N = labels.shape[0]
        grad = -labels / (probs + 1e-8)
        return grad / N

 
class MeanSquaredError(LossFunction):
    
    def __init__(self):
        pass
    
    def loss(self, probs, labels):
        loss = np.mean((probs - labels) ** 2)
        return loss
    
    def grad(self, probs, labels):
        N = labels.shape[0]
        grad = 2 * (probs - labels) / N
        return grad