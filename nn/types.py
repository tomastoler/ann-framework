class Layer:
    
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.t = self.__class__.__name__
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    

class LossFunction:
    
    def __init__(self):
        pass
    
    def loss(self, probs, labels):
        raise NotImplementedError
    
    def grad(self, probs, labels):
        raise NotImplementedError
    
    def accuracy(self, probs, labels):
        raise NotImplementedError


class Optimizer:
    
    def __init__(self):
        pass
    
    def update(self, model):
        raise NotImplementedError
    
    
class Callback:
    
    def __init__(self):
        pass
    
    def on_epochs_begin(self):
        raise NotImplementedError
    
    def on_epochs_end(self):
        raise NotImplementedError