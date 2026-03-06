# Transformer Encoder From Scratch

Projeto desenvolvido para o laboratório da disciplina **Tópicos em Inteligência Artificial**.

O objetivo deste trabalho é implementar um **Transformer Encoder do zero** utilizando apenas:

- Python
- NumPy
- Pandas

Sem utilizar bibliotecas de deep learning como PyTorch ou TensorFlow.

---

# Estrutura do projeto

Arquivos principais:

- `main.py` → executa o programa
- `attention.py` → implementação da Self Attention
- `ffn.py` → implementação da Feed Forward Network
- `encoder.py` → implementação do Encoder com 6 camadas
- `utils.py` → funções auxiliares (softmax e layer normalization)

---

# Como executar

1. Instalar as bibliotecas necessárias
pip install numpy pandas


2. Executar o programa
python main.py


---

# Saída esperada

O programa irá mostrar os shapes dos tensores durante a execução, por exemplo:
