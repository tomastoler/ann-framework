class Layer:
    
    def __init__(self):
        self.params = {}
        self.grads = {}
    
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