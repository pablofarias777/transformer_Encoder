import numpy as np
from attention import SelfAttention
from ffn import FeedForward
from utils import layer_norm


class Encoder:

    def __init__(self, d_model, num_layers=6):

        self.num_layers = num_layers
        self.attention_layers = []
        self.ffn_layers = []

        for _ in range(num_layers):
            self.attention_layers.append(SelfAttention(d_model))
            self.ffn_layers.append(FeedForward(d_model))

    def forward(self, X):

        for i in range(self.num_layers):

            X_att = self.attention_layers[i].forward(X)

            X_norm1 = layer_norm(X + X_att)
            X_ffn = self.ffn_layers[i].forward(X_norm1)
            X_out = layer_norm(X_norm1 + X_ffn)

            X = X_out

        return X