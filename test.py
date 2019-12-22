from network import Network
import numpy as np

"""
XOR test
"""
nn = Network([2,2,1],'tanh')
for i in range(5000):
    nn.train(np.array([0,0]),np.array([0]))
    nn.train(np.array([1,0]),np.array([1]))
    nn.train(np.array([0,1]),np.array([1]))
    nn.train(np.array([1,1]),np.array([0]))
print(nn.predict(np.array([0,0])))
print(nn.predict(np.array([1,0])))
print(nn.predict(np.array([0,1])))
print(nn.predict(np.array([1,1])))
