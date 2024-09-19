import numpy as np
from nn.types import Layer


class ReLU(Layer):
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad):
        return grad * (self.inputs > 0)


class Sigmoid(Layer):
    
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad):
        return grad * (self.outputs * (1 - self.outputs))


class Softmax(Layer):
    
    def forward(self, inputs):
        self.inputs = inputs
        exp_shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.outputs

    def backward(self, grad):
        batch_size, num_classes = self.outputs.shape
        dinputs = np.empty_like(grad)
        
        for i in range(batch_size):
            y = self.outputs[i].reshape(-1, 1)
            grad_i = grad[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dinputs[i] = np.dot(jacobian, grad_i).flatten()
        return dinputs

