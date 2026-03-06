import numpy as np
from utils import softmax

class SelfAttention:

    def __init__(self, d_model):

        self.d_model = d_model

        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)

    def forward(self, X):

        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        scores = Q @ K.transpose(0,2,1)

        dk = self.d_model
        scores = scores / np.sqrt(dk)

        attention_weights = softmax(scores)

        output = attention_weights @ V

        return output