import numpy as np


class SGD:
    
    def __init__(self, learning_rate: int = 0.001):
        self.learning_rate = learning_rate
        
    def update(self, model):
        for param, grad in model.get_params_and_grads():
            param -= self.learning_rate * grad
            

class Adam:
    
    def __init__(self, learning_rate: int = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
        
    def update(self, model):
        self.t += 1
        
        for i, (param, grad) in enumerate(model.get_params_and_grads()):
            if i not in self.m:
                self.m[i] = np.zeros_like(grad)
                self.v[i] = np.zeros_like(grad)
            
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)