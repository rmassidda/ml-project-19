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
hp = [{
    'minibatch': [1,16,32],
    'eta': [1e-1,1e-2],
    'topology': [[17,1,1],[17,2,1],[17,4,1]],
    }]

# Validation
val = Validation('MCL',workers=par_deg,verbose=False)

tr_loss, vl_loss, model = val.model_selection(hp,train_x,train_y,0)
plt.plot(tr_loss, color="green", label='TR')
plt.plot(vl_loss, color="red", label='VL')
plt.show()
tr_loss, te_loss, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
plt.plot(tr_loss, color="green", label='TR')
plt.plot(te_loss, color="blue", label='TE')
plt.show()
print('Hold-out produced', model, risk)

tr_loss, vl_loss, model = val.model_selection(hp,train_x,train_y,5)
plt.plot(tr_loss, color="green", label='TR')
plt.plot(vl_loss, color="red", label='VL')
plt.show()
tr_loss, vl_loss, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
print('Cross-Validation produced', model, risk)

# double_cv = val.double_cross_validation(train_x,train_y,hp,5,5)
# print('Double Cross-Validation risk estimation', double_cv)
