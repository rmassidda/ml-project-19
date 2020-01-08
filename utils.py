import numpy as np
import numpy.linalg as la

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

def square_loss(y1,y2):
    return np.dot(y1-y2,y1-y2)

def absolute_loss(y1,y2):
    return np.absolute(y1-y2)

def misclassification_loss(y1,y2,threshold=0.5):
    return (y1 > threshold) != (y2 > threshold)

def euclidian_loss(y1,y2):
    return la.norm(y1-y2,2) 

loss_dict = {
    'MSE': square_loss,
    'MEE': euclidian_loss,
    'MCL': misclassification_loss
}