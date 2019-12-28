import numpy as np
from utils import shuffle
from activation import act_dict

class Network:
    """An artificial neural network based on the multilayer perceptron architecture.

    Parameters
    ------
    topology : list
        Indicates the number of layers and units in the network.
        The i-th value in the list represents the number of units
        in the i-th layer of the network.
    activation: str, optional, default: 'sigmoid'
        The activation function used in the network.
    eta: float, optional, default: 1e-2
        The learning rate used in weight updates.
    minibatch: int, optional, default: 32
        The batch size used in the minibatch gradient descent algorithm.
    epochs: int, optional, default: 1000
        The number of epochs of the gradient descent algorithm.

    Attributes
    ------
    n_layers: int
        The number of layers in the network.
    weights: list
        The i-th element of the list represents the weight matrix
        of the i-th layer.
    biases: list
        The i-th element of the list represents the bias vector
        of the i-th layer.
    """

    def __init__(self, topology, activation='sigmoid', eta=1e-2, minibatch=32, epochs=1000):
        self.topology = topology
        self.n_layers = len(topology)
        self.eta = eta
        self.minibatch = minibatch
        self.activation = activation
        self.epochs = epochs
        self.weights = None
        self.biases = None
        self.init_weights()
        self.init_biases()

    def init_weights(self):
        """Initializes weights matrices for each layer."""
        self.weights = []
        for i in range(self.n_layers - 1):
            shape = self.topology[i+1], self.topology[i]
            self.weights.append(np.random.uniform(-1, 1, shape)) # TODO: extend to other inits

    def init_biases(self):
        """Initializes biases vectors for each layer."""
        self.biases = []
        for i in range(self.n_layers - 1):
            self.biases.append(np.random.uniform(-1, 1, self.topology[i+1])) # TODO: extend to other inits

    def predict(self, x):
        """Predicts outputs."""
        f = act_dict[self.activation]
        for i in range(self.n_layers - 1):
            x = np.dot(self.weights[i], x) + self.biases[i]
            x = f(x)
        return x

    def train(self, x, y):
        """Trains a neural network on a dataset."""
        for epoch in range(self.epochs):
            # Shuffle dataset in each epoch to avoid order bias
            x, y = shuffle(x, y)
            # Iterate over batches
            for batch_idx in range(0, len(x), self.minibatch):
                x_batch = x[batch_idx:batch_idx + self.minibatch]
                y_batch = y[batch_idx:batch_idx + self.minibatch]
                grad_w = [np.zeros(w.shape) for w in self.weights]
                grad_b = [np.zeros(b.shape) for b in self.biases]
                # Iterate over patterns
                # TODO: vectorize
                for p in range(len(x_batch)):
                    # Do a forward pass and backpropagate the error
                    nets, activations = self.forward_pass(x_batch[p])
                    grad_bp, grad_wp = self.backpropagate(nets, activations, y_batch[p])
                    # Accumulate gradients
                    grad_b = [gb+gbp for gb, gbp in zip(grad_b, grad_bp)]
                    grad_w = [gw+gwp for gw, gwp in zip(grad_w, grad_wp)]
                self.step(grad_b, grad_w, len(x_batch))
            # loss = self.compute_loss(x, y)
            # print("Epoch: %d Loss: %f" % (epoch + 1, loss))

    def compute_mse(self, x, y):
        loss = 0
        # TODO: vectorize
        for i in range(len(x)):
            pred_y = self.predict(x[i])
            loss += (pred_y - y[i])**2
        return loss / len(x)

    def compute_misclassified(self, x, y):
        err = 0
        for i in range(len(x)):
            if (y[i] > 0.5) != (self.predict(x[i]) > 0.5):
                err += 1
        return err / len(x)

    def forward_pass(self, x):
        """Computes a forward pass and returns nets and activations for each layer."""
        # TODO: different act functions for different layers
        f = act_dict[self.activation]
        nets = []
        activations = [x]
        for i in range(len(self.topology) - 1):
            x = np.dot(self.weights[i], x) + self.biases[i]
            nets.append(x)
            x = f(x)
            activations.append(x)
        return nets, activations

    def backpropagate(self, nets, activations, y):
        """Backpropagates output errors and computes gradients for each weight and bias."""
        f = act_dict[self.activation]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        # TODO: extend to other cost functions?
        delta = (y - activations[-1]) * f(nets[-1], True)
        # Compute gradients for the output layer
        grad_b[-1] = delta
        grad_w[-1] = np.outer(delta, activations[-2])
        for l in range(2, self.n_layers):
            # Compute gradients for the hidden layers
            delta = np.dot(self.weights[-l+1].T, delta) * f(nets[-l], True)
            grad_b[-l] = delta
            grad_w[-l] = np.outer(delta, activations[-l-1])
        return grad_b, grad_w

    def step(self, grad_b, grad_w, batch_size):
        """Updates weights and biases."""
        self.weights = [w + (self.eta / batch_size) * gw
                        for w, gw in zip(self.weights, grad_w)]
        self.biases = [b + (self.eta / batch_size) * gb
                       for b, gb in zip(self.biases, grad_b)]
