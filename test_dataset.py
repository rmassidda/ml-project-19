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
activations = ['tanh', 'identity']
batch_size = 32
hidden_units = 18
eta = 0.001
weight_decay = 0
# Set epochs as None to activate early stopping
epochs = None

# Training
nn = Network([n_inputs, hidden_units, n_outputs], activations=activations, minibatch=batch_size,
             eta=eta, weight_decay=weight_decay, epochs=epochs)
tr_losses, val_losses, best_epoch = nn.train(tr_x, tr_y, val_x, val_y, verbose=True)

# Plots
print("Best epoch: ", best_epoch)
plt.plot(val_losses, color="green")
plt.plot(tr_losses, color="red")
plt.show()
