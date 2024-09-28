from nn.types import Layer
from nn.losses import CrossEntropy, MeanSquaredError, CategoricalCrossEntropy
from nn.optimizers import SGD, Adam


class Sequential:
    
    def __init__(self, *, layers: list[Layer] = [], loss: str = 'crossentropy', optimizer: str = 'sgd', metrics: list[str] = ['loss'], callbacks: list[str] = []):
        self.layers = layers
        self.metrics = metrics
        match loss:
            case 'crossentropy':
                self.loss = CrossEntropy()
            case 'categorical_crossentropy':
                self.loss = CategoricalCrossEntropy()
            case 'mse':
                self.loss = MeanSquaredError()
            case _:
                raise NotImplementedError
            
        match optimizer:
            case 'sgd':
                self.optimizer = SGD()
            case 'adam':
                self.optimizer = Adam()
            case _:
                raise NotImplementedError
            
    def add(self, layer: Layer):
        self.layers.append(layer)
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                yield param, layer.grads[name]
    
    def train(self, X_train, y_train, epochs: int = 1000, learning_rate: float | None = None):
        
        if not learning_rate is None:
            self.optimizer.learning_rate = learning_rate
        
        for epoch in range(epochs + 1):
            outputs = self.forward(X_train)
            loss = self.loss.loss(outputs, y_train)
            
            grad_loss = self.loss.grad(outputs, y_train)
            self.backward(grad_loss)
            
            self.optimizer.update(self)
        
            if epoch % 100 == 0:
                match self.metrics:
                    case ['loss']:
                        print(f'Epoch {epoch}, Loss: {loss}')
                    case ['accuracy']:
                        print(f'Epoch {epoch}, Accuracy: {self.loss.accuracy(outputs, y_train)}')
                
    def predict(self, inputs):
        return self.forward(inputs)