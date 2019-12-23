from network import Network
import numpy as np

"""
Single train test
"""
nn = Network([2,2,1],'tanh')
for i in range(5000):
    nn.train_single(np.array([0,0]),np.array([0]))
    nn.train_single(np.array([1,0]),np.array([1]))
    nn.train_single(np.array([0,1]),np.array([1]))
    nn.train_single(np.array([1,1]),np.array([0]))
print("Single train")
print(nn.predict(np.array([0,0])))
print(nn.predict(np.array([1,0])))
print(nn.predict(np.array([0,1])))
print(nn.predict(np.array([1,1])))
print("------------")

"""
Minibatch test
"""
nn = Network([2,2,1],activation='tanh',minibatch=3)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
for i in range(10):
    x = np.concatenate((x,x))
    y = np.concatenate((y,y))
nn.train(x,y)
print("Mini-batch")
print(nn.predict(np.array([0,0])))
print(nn.predict(np.array([1,0])))
print(nn.predict(np.array([0,1])))
print(nn.predict(np.array([1,1])))
print("----------")
