import numpy as np
from activation import act_dict

class Network:
    # topology      list containing the number of units per layer
    # activation    name of the activation function
    # eta           step size
    def __init__(self,topology,activation,eta=0.05):
        self.layers = []
        self.eta = eta
        self.activation = activation
        for i in range(1,len(topology)):
            self.layers.append(Layer(topology[i-1],topology[i],activation))

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x, y):
        out = self.predict(x)
        sigma = y - out
        for layer in reversed(self.layers):
            f_prime = np.array([act_dict[self.activation](u.net,True) for u in layer.units])
            delta = sigma * f_prime
            # Update the weights for each unit in the layer
            for j in range(len(layer.units)):
                layer.units[j].w  = layer.units[j].w + self.eta * delta[j] * layer.units[j].x
            # Preprocessing of sigma
            sigma = delta @ np.array([u.w[1:] for u in layer.units])

class Layer:
    # n number of units in the previous layer
    # m number of units in the layer
    def __init__(self, n, m, activation ):
        self.units = [Unit(1+n, activation) for i in range(m)]

    def forward(self,x):
        return np.array([u.solicit(np.append([1],x)) for u in self.units])
    
class Unit:
    def __init__(self,k,activation):
        # Weight initialization
        self.w = np.random.rand(k)
        # Activation function
        self.activation = activation

    def solicit(self, x):
        self.x = x
        self.net = np.dot(self.x,self.w)
        return act_dict[self.activation](self.net)
