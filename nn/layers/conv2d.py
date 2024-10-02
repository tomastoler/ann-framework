import numpy as np
from nn.types import Layer


class Conv2D(Layer):

    def __init__(self, num_filters: int, filter_size: tuple[int , int], stride: int = 1, padding: int = 0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / filter_size ** 2
        self.grads = np.zeros_like(self.filters)

    def _add_padding(self, inputs):
        if self.padding == 0:
            return inputs
        padded_input = np.pad(inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        return padded_input

    def _convolve(self, inputs, kernel):
        height, width = inputs.shape[1:3]
        kernel_size = kernel.shape[0]
        output_height = (height - kernel_size) // self.stride + 1
        output_width = (width - kernel_size) // self.stride + 1
        output = np.zeros((inputs.shape[0], output_height, output_width))
        
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = inputs[:, i * self.stride: i * self.stride + kernel_size,
                               j * self.stride: j * self.stride + kernel_size]
                output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))
        
        return output
    
    def forward(self, inputs):
        batch_size, height, width, channels = inputs.shape
        inputs = self._add_padding(inputs)
        kernel_size = self.filter_size
        output_height = (height - kernel_size) // self.stride + 1
        output_width = (width - kernel_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        for f in range(self.num_filters):
            self.output[:, f, :, :] = self._convolve(inputs, self.filters[f])
        
        return self.output
    
    def backward(self, grad):
        batch_size, _, output_height, output_width = grad.shape
        d_input = np.zeros_like(self.inputs)

        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    region = self.inputs[:, i * self.stride: i * self.stride + self.filter_size,
                                        j * self.stride: j * self.stride + self.filter_size]
                    
                    self.grads[f] += np.sum(region * grad[:, f, i, j][:, None, None, None], axis=0)
                    
                    d_input[:, i * self.stride: i * self.stride + self.filter_size,
                            j * self.stride: j * self.stride + self.filter_size] += self.filters[f] * grad[:, f, i, j][:, None, None, None]

        
        if self.padding != 0:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return d_input


class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, height, width, channels = inputs.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, output_height, output_width, channels))

        for i in range(0, output_height):
            for j in range(0, output_width):
                region = inputs[:, i * self.stride:i * self.stride + self.pool_size,
                                j * self.stride:j * self.stride + self.pool_size, :]
                output[:, i, j, :] = np.max(region, axis=(1, 2))

        return output

    def backward(self, grad):
        d_input = np.zeros_like(self.input)
        batch_size, height, width, channels = self.input.shape
        
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        for i in range(output_height):
            for j in range(output_width):
                region = self.input[:, 
                                    i * self.stride:i * self.stride + self.pool_size,
                                    j * self.stride:j * self.stride + self.pool_size, :]
                
                max_vals = np.max(region, axis=(1, 2), keepdims=True)
                
                mask = (region == max_vals)
                
                d_input[:, 
                        i * self.stride:i * self.stride + self.pool_size,
                        j * self.stride:j * self.stride + self.pool_size, :] += mask * grad[:, i, j, :][:, None, None, :]
        
        return d_input