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
tr_x, tr_y, val_x, val_y = x[:bound], y[:bound], x[bound:], y[bound:]

# Net params
activations = None
f_hidden = 'tanh'
f_output = 'identity'
batch_size = 32
hidden_units = 18
eta = 0.001
weight_decay = 0
epochs = 300
tol = None
patience = 10

# Training
nn = Network([n_inputs, hidden_units, n_outputs], activations=activations, f_hidden=f_hidden,
             f_output=f_output, minibatch=batch_size, eta=eta, weight_decay=weight_decay,
             epochs=epochs, tol=tol, patience=patience)
tr_losses, val_losses, best_epoch = nn.train(tr_x, tr_y, val_x, val_y, verbose=True, f_loss='MEE')

# Plot MSE
plt.plot(tr_losses, color='red')
plt.plot(val_losses, color='green', linewidth=2, linestyle=':')
plt.ylim(0, 5)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
