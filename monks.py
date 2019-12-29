from validation import Validation
from utils import onehot, misclassification_loss
import numpy as np
import sys

train_fp = sys.argv[1]
test_fp = sys.argv[2]
monk_ranges = [3,3,2,3,4,2]

# Data load
raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
train_y = np.genfromtxt(train_fp,usecols=(0))
raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
test_y = np.genfromtxt(test_fp,usecols=(0))

# Hyperparameters
hp = [{
    'minibatch': [1,16,32],
    'eta': [1e-1,1e-2,1e-4],
    'topology': [[17,1,1],[17,2,1],[17,4,1],[17,8,1]]
    }]

# Validation
val = Validation(train_x,train_y,misclassification_loss)
model = val.model_selection(hp)
risk = val.model_assessment(model,test_x,test_y)
