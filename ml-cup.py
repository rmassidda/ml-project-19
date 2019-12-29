from validation import Validation
from utils import onehot, square_loss
import numpy as np
import sys

train_fp = sys.argv[1]

# Data load
train_x = np.genfromtxt(train_fp,delimiter=',',usecols=range(1,21))
train_y = np.genfromtxt(train_fp,delimiter=',',usecols=(21,22))

# Hyperparameters
hp = [{
    'minibatch': [16],
    'eta': [1e-2,1e-4],
    'topology': [[20,2,2],[20,4,2],[20,3,3,2]]
    }]

# Validation
val = Validation(train_x,train_y,square_loss)
model = val.model_selection(hp)
