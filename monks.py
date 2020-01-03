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
    'eta': [1e-1,1e-2],
    'topology': [[17,1,1],[17,2,1],[17,4,1]],
    'epochs': [500]
    }]

# Validation
val = Validation(misclassification_loss,verbose=False)

model_ho = val.model_selection(hp,train_x,train_y,0)
risk_ho = val.estimate_test(model_ho,train_x,train_y,test_x,test_y)
print('Hold-out produced', model_ho, risk_ho)

model_cv = val.model_selection(hp,train_x,train_y,5)
risk_cv = val.estimate_test(model_cv,train_x,train_y,test_x,test_y)
print('Cross-Validation produced', model_cv, risk_cv)

double_cv = val.double_cross_validation(train_x,train_y,hp,5,5)
print('Double Cross-Validation risk estimation', double_cv)
