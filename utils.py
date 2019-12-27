import numpy as np

def shuffle(a, b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]