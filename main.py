from nn.models import Sequential
from nn.layers import Dense
from nn.activations import ReLU, Sigmoid, Softmax
from dataset import generate_spiral_dataset

import numpy as np

X, y = generate_spiral_dataset(num_classes=3, num_points=200)

model = Sequential(
    layers=[
        Dense(2, 6),
        ReLU(),
        Dense(6, 12),
        Sigmoid(),
        Dense(12, 9),
        ReLU(),
        Dense(9, 3),
        Softmax()
    ],
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    # callbacks=['accuracy']
)

model.train(X, y, epochs=1000)

n = int(input(f"Enter number of points to predict (0 - {len(X)}) -> "))
print(f"X -> {X[n]}")
print(f"y -> {y[n]}")
print(f"y_hat -> {np.argmax(model.predict(X[n]))} | {model.predict(X[n])}")
