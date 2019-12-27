import numpy as np

def tanh(x, diff=False):
    return 1 - np.tanh(x)**2 if diff else np.tanh(x)

def sigmoid(x, diff=False):
    sigmoid = (1 / (1 + np.exp(-x)))
    return sigmoid * (1 - sigmoid) if diff else sigmoid

act_dict = {
    'tanh': tanh,
    'sigmoid': sigmoid
}
