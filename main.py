from nn.models import Sequential
from nn.layers import Dense, Flatten
from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.callbacks import Timer
from dataset import generate_spiral_dataset

import keras
import numpy as np

X, y = generate_spiral_dataset(num_classes=3, num_points=200)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# X_train = X_train.reshape(60000, 28*28)
# X_test = X_test.reshape(10000, 28*28)
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

model = Sequential(
    layers=[
        Flatten(),
        Dense(28*28, 128),
        ReLU(),
        Dense(128, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ],
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.train(X_train, y_train, epochs=100, epochs_per_log=20)

