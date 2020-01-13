from matplotlib import pyplot as plt
from network import Network
from utils import shuffle
import pandas as pd

# Parse the dataset
ds = pd.read_csv('data/ML-CUP19-TR.csv', delimiter=',', comment='#', header=None).values
x = ds[:, 1:21]
y = ds[:, 21:]

n_inputs = x.shape[1]
n_outputs = y.shape[1]

# Shuffle the dataset
x, y = shuffle(x, y)

# TR/VL split
bound = int(len(x) * 0.75)
tr_x, tr_y, ts_x, ts_y = x[:bound], y[:bound], x[bound:], y[bound:]

# Net params
activations = None
f_hidden = 'relu'
f_output = 'identity'
batch_size = 32
hidden_units = 30
eta = 0.001
momentum = 0.9
weight_decay = 0.001
epochs = 100
tol = None
patience = 10
losses = ['MEE', 'MSE']

# Training
nn = Network([n_inputs, hidden_units, n_outputs], activations=activations, f_hidden=f_hidden,
             f_output=f_output, minibatch=batch_size, eta=eta, weight_decay=weight_decay,
             epochs=epochs, tol=tol, patience=patience, momentum=momentum)
tr_losses, ts_losses, best_epoch = nn.train(tr_x, tr_y, ts_x, ts_y, verbose=True, losses=losses)

# Plot MEE
for i, loss in enumerate(losses):
    plt.plot(tr_losses[i], color='red', label='TR')
    plt.plot(ts_losses[i], color='green', linewidth=2, linestyle=':', label='TS')
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.legend()
    plt.show()

