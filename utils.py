import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
















from utils import softmax
test = np.array([1.0, 2.0, 3.0])
print("\nTeste Softmax:")
print(softmax(test))