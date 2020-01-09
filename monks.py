from matplotlib import pyplot as plt
from utils import onehot
from validation import Validation
import numpy as np
import sys

train_fp = sys.argv[1]
test_fp = sys.argv[2]
par_deg = int(sys.argv[3])
monk_ranges = [3,3,2,3,4,2]

# Data load
raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
train_y = np.genfromtxt(train_fp,usecols=(0))
raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
test_y = np.genfromtxt(test_fp,usecols=(0))

# Hyperparameters
early_stopping = [{
    'minibatch': [1,16],
    'eta': [1e-1,1e-2],
    'topology': [[17,2,1],[17,4,1],[17,8,1]],
    'momentum': [0.9,0.99],
    'weight_decay': [1e-2,1e-4],
    'patience': [50]
    }]
fixed_epoch = [{
    'minibatch': [1,16],
    'eta': [1e-1,1e-2],
    'topology': [[17,2,1],[17,4,1],[17,8,1]],
    'momentum': [0.9,0.99],
    'weight_decay': [1e-2,1e-4],
    'epochs': [500]
    }]

hf = [early_stopping, fixed_epoch]

# Validation
val = Validation('MCL',workers=par_deg,verbose=True)

# Identify the best family of models via double cross-validation
print('Start double CV')
family_risk = np.Inf
for h in hf:
    double_cv = val.double_cross_validation(train_x,train_y,h,5,3)
    if double_cv < family_risk:
        family = h
        family_risk = double_cv
print('Chosen family:')
print(family, family_risk,end='\n\n')

# Select the best model via cross-validation
print('Model selection')
model_tr, model_vl, model = val.model_selection(family,train_x,train_y,5)
print('Chosen model:')
print(model,model_tr,model_vl,end='\n\n')

# Model assessment on the test set
print('Model assesment')
tr_loss, te_loss, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
print('Chosen model:')
print(model,risk)

# Plot of the estimation
plt.title('Risk estimation')
plt.plot(tr_loss, color="green", label='TR')
plt.plot(te_loss, color="blue", label='TE')
plt.legend()
plt.show()
