import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)



from utils import softmax
test = np.array([1.0, 2.0, 3.0])
print("\nTeste Softmax:")
print(softmax(test))


def layer_norm(X, epsilon=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    X_norm = (X - mean) / np.sqrt(var + epsilon)

    return X_norm