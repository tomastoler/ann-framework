import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def generate_spiral_dataset(num_classes, num_points, noise=0.2):
    X = []
    y = []
    
    for class_number in range(num_classes):
        r = np.linspace(0.0, 1, num_points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, num_points) + np.random.randn(num_points) * noise
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.c_[x1, x2])
        y.append(np.full(num_points, class_number))
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    y_one_hot = one_hot_encode(y, num_classes)
    
    return X, y_one_hot

def generate_functional_dataset(num_points):
    X = np.random.randn(num_points, 1)
    y = np.cos(X)
    return X, y

if __name__ == '__main__':
    num_classes = 6
    num_points = 200

    X, y_one_hot = generate_spiral_dataset(num_classes, num_points)

    print("X shape:", X.shape)
    print("y_one_hot shape:", y_one_hot.shape)

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y_one_hot, axis=1), cmap='jet', s=40)
    plt.title(f'Spiral Dataset with {num_classes} Classes')
    plt.show()
