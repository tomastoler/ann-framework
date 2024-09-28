import numpy as np
from nn.types import LossFunction


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
    

class MeanAbsoluteError(LossFunction):
    
    def __init__(self):
        pass
    
    def loss(self, probs, labels):
        loss = np.mean(np.abs(probs - labels))
        return loss
    
    def grad(self, probs, labels):
        N = labels.shape[0]
        grad = 2 * (probs - labels) / N
        return grad


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
    
    def accuracy(self, probs, labels):
        return np.mean(np.argmax(probs, axis=1) == np.argmax(labels, axis=1))


class CategoricalCrossEntropy(LossFunction):
    
    def __init__(self):
        self.epsilon = 1e-15 

    def loss(self, probs, labels):
        probs = np.clip(probs, self.epsilon, 1. - self.epsilon)
        loss = -np.sum(labels * np.log(probs), axis=1)
        
        return np.mean(loss)

    def grad(self, probs, labels):
        grad = probs - labels
        return grad / probs.shape[0]
    
    def accuracy(self, probs, labels):
        return np.mean(np.argmax(probs, axis=1) == np.argmax(labels, axis=1))
    

class BinaryCrossEntropy(LossFunction):
    
    def __init__(self):
        self.epsilon = 1e-15
        
    def loss(self, probs, labels):
        probs = np.clip(probs, self.epsilon, 1. - self.epsilon)
        loss = -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs), axis=1)
        
        return np.mean(loss)
    
    def grad(self, probs, labels):
        grad = probs - labels
        return grad / probs.shape[0]
    
    def accuracy(self, probs, labels):
        return np.mean(np.argmax(probs, axis=1) == np.argmax(labels, axis=1))
