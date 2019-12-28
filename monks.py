from network import Network
from utils import onehot
from grid import Grid
import numpy as np
import sys

train_fp = sys.argv[1]
test_fp = sys.argv[2]
monk_ranges = [3,3,2,3,4,2]

# Data load
raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
train_y = np.genfromtxt(train_fp,usecols=(0))

# Hyperparameters
hp = [{
        'minibatch': [1,16,32],
        'eta': [1e-1,1e-2,1e-4],
        'hidden_units': [1,2,4,8]
    }]

# Model selection
grid = Grid(hp)
k = 5
risk = np.Inf
best = None
for p in grid:
    nn = Network(
            [17, p['hidden_units'], 1],
            activation='sigmoid',
            eta=p['eta'],
            minibatch=p['minibatch'],
            epochs=500)

    # Split with validation set (75/25)
    bound = int(len(train_x)*(3/4))
    nn.train(train_x[:bound],train_y[:bound])

    valid_x = train_x[bound:]
    valid_y = train_y[bound:]

    estimated_risk = nn.compute_misclassified(valid_x,valid_y)
    print(p, estimated_risk,sep='\t')
    if estimated_risk < risk:
        best = p
        risk = estimated_risk

# Model assestment
nn = Network(
        [17, best['hidden_units'], 1],
        activation='sigmoid',
        eta=best['eta'],
        minibatch=best['minibatch'],
        epochs=500)
# Train on the best model
nn.train(train_x,train_y)
raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
test_y = np.genfromtxt(test_fp,usecols=(0))
estimated_risk = nn.compute_misclassified(test_x,test_y)
print("=== RISK ESTIMATION ===")
print(best, estimated_risk,sep='\t')
