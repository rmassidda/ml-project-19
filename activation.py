import numpy as np

def tanh(x,diff=False):
    return 1 - np.tanh(x)**2 if diff else np.tanh(x)

act_dict = {
    'tanh': tanh
}
