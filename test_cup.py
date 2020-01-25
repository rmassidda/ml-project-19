from matplotlib import pyplot as plt
from network import Network
from utils import shuffle, loss_dict
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
bound_ts = int(len(x) * 0.8)
bound_tr = int(len(x) * 0.6)
dev_x, dev_y, ts_x, ts_y = x[:bound_ts], y[:bound_ts], x[bound_ts:], y[bound_ts:]
tr_x, tr_y, val_x, val_y = dev_x[:bound_tr], dev_y[:bound_tr], dev_x[bound_tr:], dev_y[bound_tr:]

# Net params
params = {
    "topology": [n_inputs, 32, 2],
    "f_hidden": 'tanh',
    "minibatch": 32,
    "eta": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0,
    "patience": 100,
    "max_norm": 0,
    "prefer_tr": False,
    'tau': 10000,
    'eta_zero': 0.05
}

losses = ['MEE']

# Training
nn = Network(**params)
tr_losses, ts_losses, best_epoch = nn.train(tr_x, tr_y, val_x, val_y, verbose=True, losses=losses)

print(nn.error(ts_x, ts_y, loss_dict['MEE']))

# Plot MEE
for i, loss in enumerate(losses):
    plt.plot(tr_losses[i], color='red', label='TR')
    plt.plot(ts_losses[i], color='green', linewidth=2, linestyle=':', label='TS')
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.legend()
    plt.show()

