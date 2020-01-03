from network import Network
import pandas as pd

ds = pd.read_csv('data/ML-CUP19-TR.csv', delimiter=',', comment='#', header=None).values
x = ds[:, 1:21]
y = ds[:, 21:]

n_inputs = x.shape[1]
n_outputs = y.shape[1]

activations = ['tanh', 'identity']
batch_size = 32
hidden_units = 18
eta = 0.001
epochs = 100

nn = Network([n_inputs, hidden_units, n_outputs], activations=activations, minibatch=batch_size,
             eta=eta, epochs=epochs)
nn.train(x, y)