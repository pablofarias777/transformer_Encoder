import numpy as np
import pandas as pd

vocab = {
    "o": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartao": 3
}

df_vocab = pd.DataFrame(list(vocab.items()), columns=["palavra", "id"])

print("VOCABULÁRIO")
print(df_vocab)

sentence = ["o", "banco", "bloqueou", "cartao"]

ids = [vocab[word] for word in sentence]

print("\nIDS DA FRASE:")
print(ids)

d_model = 64

vocab_size = len(vocab)

embedding_table = np.random.randn(vocab_size, d_model)

X = embedding_table[ids]

X = np.expand_dims(X, axis=0)

print("\nShape do tensor X:")
print(X.shape)

from attention import SelfAttention

attention = SelfAttention(d_model)

X_att = attention.forward(X)

print("\nShape saída attention:")
print(X_att.shape)

from utils import layer_norm

X_res = X + X_att

X_norm = layer_norm(X_res)

print("\nShape após LayerNorm:")
print(X_norm.shape)

from ffn import FeedForward

ffn = FeedForward(d_model)

X_ffn = ffn.forward(X_norm)

print("\nShape saída FFN:")
print(X_ffn.shape)