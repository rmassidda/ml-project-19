from network import Network
from utils import onehot
import numpy as np
import sys

train_fp = sys.argv[1]
test_fp = sys.argv[2]
monk_ranges = [3,3,2,3,4,2]

batch_size = 1
nn = Network([17, 3, 1], activation='sigmoid', eta=5e-2, minibatch=batch_size,epochs=500)

raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
train_y = np.genfromtxt(train_fp,usecols=(0))
nn.train(train_x,train_y)
print ( '== TRAIN ==' )
print ( 'Misclassified', nn.compute_misclassified(train_x,train_y) )
print ( 'MSE', nn.compute_mse(train_x,train_y) )

raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
test_y = np.genfromtxt(test_fp,usecols=(0))
print ( '== TEST ==' )
print ( 'Misclassified', nn.compute_misclassified(test_x,test_y) )
print ( 'MSE', nn.compute_mse(test_x,test_y) )
