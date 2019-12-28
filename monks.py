from concurrent.futures import ThreadPoolExecutor
from grid import Grid
from network import Network
from utils import onehot
import numpy as np
import sys

def validate(par):
    nn = Network(
            [17, par['hidden_units'], 1],
            activation='sigmoid',
            eta=par['eta'],
            minibatch=par['minibatch'],
            epochs=500)

    # Split with validation set (75/25)
    bound = int(len(train_x)*(3/4))
    nn.train(train_x[:bound],train_y[:bound])

    valid_x = train_x[bound:]
    valid_y = train_y[bound:]

    estimated_risk = nn.compute_misclassified(valid_x,valid_y)
    return par,estimated_risk

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
risk = np.Inf
best = None

# Parallel grid search
print("=== MODEL SELECTION ===")
with ThreadPoolExecutor(max_workers=8) as executor:
    grid_res = executor.map(validate,grid)
    for p in grid_res:
        print(p,sep='\t')
        if p[1] < risk:
            risk = p[1]
            best = p[0]

# Train on the best model
nn = Network(
        [17, best['hidden_units'], 1],
        activation='sigmoid',
        eta=best['eta'],
        minibatch=best['minibatch'],
        epochs=500)
nn.train(train_x,train_y)

raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
test_y = np.genfromtxt(test_fp,usecols=(0))

# Model assessment
print("=== MODEL ASSESSMENT ===")
estimated_risk = nn.compute_misclassified(test_x,test_y)
print(best, estimated_risk,sep='\t')
