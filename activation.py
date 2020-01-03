import numpy as np

def identity(x, diff=False):
    return np.ones(x.shape) if diff else x

def relu(x, diff=False):
    return np.where(x < 0, 0, 1) if diff else np.where(x < 0, 0, x)

def tanh(x, diff=False):
    return 1 - np.tanh(x)**2 if diff else np.tanh(x)

def sigmoid(x, diff=False):
    sigmoid = (1 / (1 + np.exp(-x)))
    return sigmoid * (1 - sigmoid) if diff else sigmoid

act_dict = {
    'identity': identity,
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid
}