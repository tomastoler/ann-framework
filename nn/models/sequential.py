from nn.types import Layer, Optimizer, LossFunction, Callback
from nn.layers import Dense, Flatten
from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.losses import CrossEntropy, MeanSquaredError, CategoricalCrossEntropy, MeanAbsoluteError, BinaryCrossEntropy, SparseCategoricalCrossEntropy
from nn.optimizers import SGD, Adam

import json


class Sequential:
    
    def __init__(
        self,
        *,
        layers: list[Layer] = [],
        loss: str | LossFunction  = 'crossentropy',
        optimizer: str | Optimizer = 'sgd',
        metrics: list[str] = ['loss'],
        callbacks: list[str | Callback] = []
    ):
        self.layers = layers
        self.metrics = metrics
        self.callbacks = callbacks
        
        self.set_loss(loss)
        self.set_optimizer(optimizer)

    def set_loss(self, loss: str | LossFunction):
        if type(loss) == str:
            match loss:
                case 'mse':
                    self.loss = MeanSquaredError()
                case 'mae':
                    self.loss = MeanAbsoluteError()
                case 'crossentropy':
                    self.loss = CrossEntropy()
                case 'categorical_crossentropy':
                    self.loss = CategoricalCrossEntropy()
                case 'binary_crossentropy':
                    self.loss = BinaryCrossEntropy()
                case 'sparse_categorical_crossentropy':
                    self.loss = SparseCategoricalCrossEntropy()
                case _:
                    raise NotImplementedError
            self.ls = loss
        else:
            self.loss = loss
            self.ls = self.loss.t
        
    def set_optimizer(self, optimizer: str | Optimizer):
        if type(optimizer) == str:
            match optimizer:
                case 'sgd':
                    self.optimizer = SGD()
                case 'adam':
                    self.optimizer = Adam()
                case _:
                    raise NotImplementedError
            self.opt = optimizer
        else:
            self.optimizer = optimizer
            self.opt = self.optimizer.t
            
    def add(self, layer: Layer):
        self.layers.append(layer)
        
    def forward(self, inputs, training=True):
        if not training:
            for layer in self.layers:
                inputs = layer.forward(inputs)
        else:
            for layer in self.layers:
                inputs = layer.forward(inputs, training)
        return inputs
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def predict(self, inputs):
        return self.forward(inputs, training=False)
    
    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                yield param, layer.grads[name]
    
    def train(
        self,
        X_train,
        y_train,
        epochs: int = 1000,
        learning_rate: float | None = None,
        epochs_per_log: int = 100
    ):
        
        if not learning_rate is None:
            self.optimizer.learning_rate = learning_rate
        
        for epoch in range(epochs + 1):
            outputs = self.forward(X_train)
            
            self._callbacks_begin()
            
            grad_loss = self.loss.grad(outputs, y_train)
            self.backward(grad_loss)
            
            self.optimizer.update(self)
        
            if epoch % epochs_per_log == 0:
                print(f"Epoch {epoch}", end=' ')
                self._metrics(outputs, y_train)
                self._callbacks_end()
    
    def _metrics(self, probs, labels):
        for metric in self.metrics:
            if metric == 'loss':
                print(f" | Loss: {round(self.loss.loss(probs, labels), 2)}", end=' ')
            if metric == 'accuracy':
                print(f" | Accuracy: {round(self.loss.accuracy(probs, labels) * 100, 1)}%", end=' ')
        print()
        
    def _callbacks_begin(self):
        for callback in self.callbacks:
            try:
                callback.on_epochs_begin()
            except:
                continue
    
    def _callbacks_end(self):
        for callback in self.callbacks:
            try:
                callback.on_epochs_end()
            except:
                continue
    
    def save(self, path: str) -> None:
        model_data = {
            'layers': [],
            'loss': self.ls,
            'optimizer': self.opt,
            'metrics': self.metrics
        }
        
        for layer in self.layers:
            model_data['layers'].append({
                "type": layer.t,
                "params": {
                    "W": layer.params['W'].tolist() if layer.params != {} else [],
                    "B": layer.params['B'].tolist() if layer.params != {} else []
                },
                "input_size": len(layer.params['W']) if layer.params != {} else 0,
                "output_size": len(layer.params['B']) if layer.params != {} else 0,
            })
            
        with open(path, 'w') as f:
            json.dump(model_data, f)
            
    @classmethod
    def load(cls, path: str) -> 'Sequential':
        with open(path, 'r') as f:
            model_data = json.load(f)
            
        model = cls(
            layers=[],
            loss=model_data['loss'],
            optimizer=model_data['optimizer'],
            metrics=model_data['metrics']
        )
        
        existing_layers = {
            'Dense': Dense,
            'Flatten': Flatten,
            'Softmax': Softmax,
            'ReLU': ReLU,
            'Sigmoid': Sigmoid,
            'Tanh': Tanh
        }
        
        for layer_data in model_data['layers']:
            layer = existing_layers[layer_data['type'].capitalize()].from_json(layer_data)
            model.add(layer)
            
        return model