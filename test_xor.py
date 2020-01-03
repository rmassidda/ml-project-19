from network import Network
import numpy as np

# XOR test

activations = ['tanh', 'sigmoid']
batch_size = 3
hidden_units = 10
eta = 0.6
epochs = 5000

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
if activations[-1] == 'tanh':
    # 0 encoded as -1 because of tanh
    y = np.array([[-1], [1], [1], [-1]])
elif activations[-1] == 'sigmoid':
    y = np.array([[0], [1], [1], [0]])

nn = Network([2, hidden_units, 1],  activations=activations, minibatch=batch_size,
             eta=eta, epochs=epochs)
nn.train(x, y)

print('------')
print('XOR predictions:')
print('0 0 -->', nn.predict(np.array([0, 0])))
print('0 1 -->', nn.predict(np.array([0, 1])))
print('1 0 -->', nn.predict(np.array([1, 0])))
print('1 1 -->', nn.predict(np.array([1, 1])))
