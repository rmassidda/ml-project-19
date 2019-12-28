import numpy as np

def shuffle(a, b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]

def onehot(x,ranges):
    h = np.zeros(sum(ranges))
    s = 0
    for r in range(len(x)):
        h[int(x[r])-1+s] = 1
        s += ranges[r]
    return h
