import numpy as np
from nn.types import Layer


class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params['W'] = np.random.randn(self.input_size, self.output_size)
        self.params['B'] = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.params['W']) + self.params['B']
    
    def backward(self, grad):
        self.grads['W'] = np.dot(self.inputs.T, grad)
        self.grads['B'] = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.params['W'].T)
    
    @classmethod
    def from_json(cls, data):
        layer = cls(data['input_size'], data['output_size'])
        layer.params = data['params']
        return layer
    

class Flatten(Layer):
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = self.input_shape[0]
        return inputs.reshape(batch_size, -1)
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)
    
    @classmethod
    def from_json(cls, data):
        layer = cls()
        return layer