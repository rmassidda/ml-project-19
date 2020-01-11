from matplotlib import pyplot as plt
from network import Network
from utils import shuffle, onehot
import numpy as np
import sys


# Command line arguments
if len(sys.argv) == 4:
    train_fp = sys.argv[1]
    test_fp  = sys.argv[2]
else:
    train_fp = 'data/monks/monks-1.train'
    test_fp  = 'data/monks/monks-1.test'

monk_ranges = [3,3,2,3,4,2]

# Data load
raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
tr_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
tr_y = np.genfromtxt(train_fp,usecols=(0))
raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
ts_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
ts_y = np.genfromtxt(test_fp,usecols=(0))

n_inputs = 17
n_outputs = 1

# Shuffle the datasets
tr_x, tr_y = shuffle(tr_x, tr_y)
ts_x, ts_y = shuffle(ts_x, ts_y)

# Net params
activations = None
f_hidden = 'tanh'
f_output = 'sigmoid'
batch_size = 32
hidden_units = 10
eta = 0.5
weight_decay = 0
epochs = 500
tol = None
patience = 10

# Training
nn = Network([n_inputs, hidden_units, n_outputs], activations=activations, f_hidden=f_hidden,
             f_output=f_output, minibatch=batch_size, eta=eta, weight_decay=weight_decay,
             epochs=epochs, tol=tol, patience=10)
tr_err, ts_err, best_epoch = nn.train(tr_x, tr_y, ts_x, ts_y, verbose=True, f_loss='MCL')

# Plot accuracies (percentage)
tr_acc = 100 * (1 - tr_err)
ts_acc = 100 * (1 - ts_err)
plt.plot(tr_acc, color='red')
plt.plot(ts_acc, color='green', linewidth=2, linestyle=':')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()
