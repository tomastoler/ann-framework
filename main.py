from nn.models import Sequential
from nn.layers import Dense, Flatten, Dropout
from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.callbacks import Timer
from dataset import generate_functional_dataset

X, y = generate_functional_dataset(1000)

X_train, X_test, y_train, y_test = X[:900], X[900:], y[:900], y[900:]

model = Sequential(
    layers=[
        Dense(1, 8),
        ReLU(),
        Dense(8, 12),
        ReLU(),
        Dense(12, 1)
    ],
    loss='mse',
    optimizer='adam',
)

model.train(X_train, y_train, epochs=1000, epochs_per_log=200)

