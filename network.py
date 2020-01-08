import numpy as np
from utils import shuffle, loss_dict
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
    weight_decay: float, optional, default: 1e-4
        Weight decay parameter (L2 regularization).
    momentum: float, optional, default: 0.9
        Momentum parameter.
    minibatch: int, optional, default: 32
        The batch size used in the minibatch gradient descent algorithm.
    epochs: int, optional, default: 1000
        The number of epochs to be executed in the training of the network.
    patience: int, optional, default: 20
        Patience parameter used in early stopping.

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
    V_dw: list
        Moving averages for weights (momentum gradient descent). The i-th
        element of the list represents the moving average matrix
        for the weights in the i-th layer.
    V_db: list
        Moving averages for biases (momentum gradient descent). The i-th
        element of the list represents the moving average matrix
        for the biases in the i-th layer.
    """

    def __init__(self, topology, activations=None, eta=1e-2,
                 weight_decay=1e-4, momentum=0.9, minibatch=32,
                 epochs=None, patience=20):
        self.topology = topology
        self.n_layers = len(topology)
        self.eta = eta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.minibatch = minibatch
        self.activations = activations
        self.epochs = epochs
        self.patience = patience

        self.weights = None
        self.biases = None
        self.V_dw = None
        self.V_db = None

    def initialize(self):
        """Initializes the neural network."""
        # Init weights (Glorot et al.)
        self.weights = []
        for i in range(self.n_layers - 1):
            self.weights.append(np.random.randn(self.topology[i+1], self.topology[i]) * \
                 np.sqrt(2/(self.topology[i+1]+self.topology[i])))

        # Init biases
        self.biases = []
        for i in range(self.n_layers - 1):
            self.biases.append(np.zeros(self.topology[i+1]))
        
        # Init weights moving averages
        self.V_dw = []
        for i in range(self.n_layers - 1):
            self.V_dw.append(np.zeros((self.topology[i+1], self.topology[i])))

        # Init biases moving averages
        self.V_db = []
        for i in range(self.n_layers - 1):
            self.V_db.append(np.zeros(self.topology[i+1]))

        # Default: tanh for hidden layers and identity for output layer (regression)
        if self.activations is None:
            self.activations = ['tanh'] * (self.n_layers - 1)
            self.activations[-1] = 'identity'

    def predict(self, x):
        """Predicts outputs."""
        for i in range(self.n_layers - 1):
            f = act_dict[self.activations[i]]
            x = np.dot(self.weights[i], x) + self.biases[i]
            x = f(x)
        return x

    def train(self, x, y, val_x=None, val_y=None, f_loss='MSE', verbose=False):
        """Trains a neural network on a dataset."""
        self.initialize()
        tr_losses = []
        val_losses = []

        early_stopping = True if self.epochs is None and \
                         val_x is not None and val_y is not None else False
        if early_stopping:
            no_improvement = 0
            best_val_loss = np.Inf

        epoch = 0
        training = True

        while training:

            epoch += 1

            if verbose:
                print("Epoch: %d" % epoch)
            # Shuffle dataset in each epoch to avoid order bias
            # And execute a single epoch 
            x, y = shuffle(x, y)
            self.epoch_train(x, y)

            # Compute losses on TR and VL
            tr_loss = self.error(x, y, loss_dict[f_loss])
            tr_losses.append(tr_loss)
            if verbose:
                print("\tTraining loss: %f" % tr_loss)
            if val_x is not None and val_y is not None:
                val_loss = self.error(val_x, val_y, loss_dict[f_loss])
                val_losses.append(val_loss)
                if verbose:
                    print("\tValidation loss: %f" % val_loss)

            # Check stop conditions
            if early_stopping:
                if val_loss >= best_val_loss:
                    no_improvement += 1
                else:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if no_improvement > 0:
                        no_improvement = 0 
                if no_improvement == self.patience:
                    # Stop training if the nn didn't improve on VL for patience consecutive epochs
                    training = False
            else:
                if epoch == self.epochs:
                    # Stop training when the prefixed number of epochs has been reached
                    training = False

        if not early_stopping:
            best_epoch = epoch

        return tr_losses, val_losses, best_epoch - 1

    def epoch_train(self, x, y):
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

    def error(self, x, y, loss):
        err = 0
        for i in range(len(x)):
            err += loss(self.predict(x[i]),y[i])
        return err / len(x)

    def forward_pass(self, x):
        """Computes a forward pass and returns nets and activations for each layer."""
        nets = []
        activations = [x]
        for i in range(self.n_layers - 1):
            f = act_dict[self.activations[i]]
            x = np.dot(self.weights[i], x) + self.biases[i]
            nets.append(x)
            x = f(x)
            activations.append(x)
        return nets, activations

    def backpropagate(self, nets, activations, y):
        """Backpropagates output errors and computes gradients for each weight and bias."""
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        f = act_dict[self.activations[-1]]
        # TODO: extend to other cost functions?
        delta = (y - activations[-1]) * f(nets[-1], True)
        # Compute gradients for the output layer
        grad_b[-1] = delta
        grad_w[-1] = np.outer(delta, activations[-2])
        for l in range(2, self.n_layers):
            f = act_dict[self.activations[-l]]
            # Compute gradients for the hidden layers
            delta = np.dot(self.weights[-l+1].T, delta) * f(nets[-l], True)
            grad_b[-l] = delta
            grad_w[-l] = np.outer(delta, activations[-l-1])
        return grad_b, grad_w

    def step(self, grad_b, grad_w, batch_size):
        """Updates weights and biases."""
        # Update moving averages
        self.V_dw = [self.momentum * v_dw + (1 - self.momentum) * gw
                     for v_dw, gw in zip(self.V_dw, grad_w)]
        self.V_db = [self.momentum * v_db + (1 - self.momentum) * gb
                     for v_db, gb in zip(self.V_db, grad_b)]
        
        # Update weights and biases using momentum and L2 regularization
        self.weights = [w + (self.eta / batch_size) * v_dw - self.weight_decay * w
                        for w, v_dw in zip(self.weights, self.V_dw)]
        self.biases = [b + (self.eta / batch_size) * v_db
                       for b, v_db in zip(self.biases, self.V_db)]
