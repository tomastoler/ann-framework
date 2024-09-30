import numpy as np
from nn.types import LossFunction


class MeanSquaredError(LossFunction):
    
    t = 'mse'
    
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
    
    t = 'mae'
    
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
    
    t = 'crossentropy'
    
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
    
    t = 'categorical_crossentropy'
    
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
    
    t = 'binary_crossentropy'
    
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


class SparseCategoricalCrossEntropy(LossFunction):
    
    t = 'sparse_categorical_crossentropy'
    
    def __init__(self):
        pass
    
    def loss(self, probs, labels):
        batch_size = probs.shape[0]
        correct_class_prob = probs[np.arange(batch_size), labels]
        loss = -np.log(correct_class_prob + 1e-9)

        return np.mean(loss)

    def grad(self, probs, labels):
        grad = probs.copy()
        batch_size = probs.shape[0]
        grad[np.arange(batch_size), labels] -= 1
        grad /= batch_size
        
        return grad
    
    def accuracy(self, probs, labels):
        return np.mean(np.argmax(probs, axis=1) == labels)