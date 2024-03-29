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
    activations: list, optional, default: None
        A list that contains the activation functions used in the
        network. The i-th activation function in the list will be
        used in the (i+1)-st layer of the network (the input layer
        is not considered). If set to a value, f_hidden and f_output
        will be ignored.
    f_hidden: str, optional, default: 'tanh'
        Activation function used for each hidden layer.
    f_output: str, optional, default: 'identity'
        Activation function used in the output layer.
    eta: float, optional, default: 1e-2
        The fixed learning rate used in weight updates.
    weight_decay: float, optional, default: 1e-4
        Weight decay parameter (L2 regularization).
    momentum: float, optional, default: 0.9
        Momentum parameter.
    minibatch: int, optional, default: 32
        The batch size used in the minibatch gradient descent algorithm.
    epochs: int, optional, default: None
        If set to a value, training stops when this number of epochs is
        reached.
    tol: float, optional, default: 0.0
        If set to a value, training stops when loss on TR has not improved
        by at least tol for patience consecutive epochs.
    patience: int, optional, default: 20
        Patience parameter used in early stopping mode
    max_norm: float, optional, default: 1
        Norm scaling to avoid gradient explosion
    tau: float, optional, default: 1
        Number of iterations in which eta decays linearly,
        after tau iterations the learning rate is fixed
    eta_zero: float optional, default: 5e-1
        Initial learning rate to be decreased linearly
        for tau iterations up to use fixed eta

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

    def __init__(self, topology, activations=None, f_hidden='tanh',
                 f_output='identity', eta=1e-2, weight_decay=1e-4,
                 momentum=0.9, minibatch=32, epochs=None, tol=0.0,
                 patience=20, max_epochs=3000, max_norm=1,
                 tau=1, eta_zero=5e-1, prefer_tr=True):
        self.topology = topology
        self.activations = activations
        self.f_hidden = f_hidden
        self.f_output = f_output
        self.eta = eta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.minibatch = minibatch
        self.epochs = epochs
        self.tol = tol
        self.patience = patience
        self.max_epochs = max_epochs
        self.max_norm = max_norm
        self.tau = tau
        self.eta_zero = eta_zero
        self.eta_tau  = eta
        self.prefer_tr = prefer_tr

        self.n_layers = len(topology)
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

        # If not already set, initialize activations 
        if self.activations is None:
            self.activations = [self.f_hidden] * (self.n_layers - 1)
            self.activations[-1] = self.f_output

        # Number of minibatch iterations
        self.k = 1

    def predict(self, x):
        """Predicts outputs."""
        for i in range(self.n_layers - 1):
            f = act_dict[self.activations[i]]
            x = np.dot(self.weights[i], x) + self.biases[i]
            x = f(x)
        return x

    def train(self, x, y, val_x=None, val_y=None, losses=['MSE'], verbose=False):
        """Trains a neural network on a dataset."""
        self.initialize()
        tr_losses = np.empty((len(losses),self.max_epochs+1))
        vl_losses = np.empty((len(losses),self.max_epochs+1))

        vl_stop = True if self.epochs is None and val_x is not None and val_y is not None \
                and not self.prefer_tr else False
        tr_stop = True if self.epochs is None and not vl_stop else False
        early_stop = vl_stop or tr_stop

        no_improvement = 0
        best_loss = np.Inf
        epoch = 0
        training = True

        while training:

            # Training
            self.epoch_train(x, y)

            if verbose:
                print('Epoch: %d' % epoch)
                print('Eta: ',self.eta)

            # Compute losses on the training set
            for i, loss in enumerate(losses):
                tr_losses[i][epoch] = self.error(x, y, loss_dict[loss])
                if verbose:
                    print('\tTraining loss (%s): %f' % (loss, tr_losses[i][epoch]))

            # Compute losses on the validation set
            if val_x is not None and val_y is not None:
                for i, loss in enumerate(losses):
                    vl_losses[i][epoch] = self.error(val_x, val_y, loss_dict[loss])
                    if verbose:
                        print('\tValidation loss (%s): %f' % (loss, vl_losses[i][epoch]))

            # Check stop conditions
            if early_stop:
                if vl_stop:
                    curr_loss = vl_losses[0][epoch]
                    if verbose:
                        print('VL',epoch, curr_loss, best_loss)
                else:
                    curr_loss = tr_losses[0][epoch]
                    if verbose:
                        print('TR',epoch, curr_loss, best_loss)

                if curr_loss >= best_loss - self.tol:
                    no_improvement += 1
                else:
                    no_improvement = 0

                if curr_loss <= best_loss: 
                    best_loss  = curr_loss
                    best_epoch = epoch

                if no_improvement >= self.patience or epoch == self.max_epochs:
                    training = False
            elif epoch >= self.epochs:
                # Stop training when the prefixed number of epochs has been reached
                training = False

            epoch += 1

        if not early_stop:
            best_epoch = epoch - 1

        return tr_losses[:,:best_epoch+1], vl_losses[:,:best_epoch+1], best_epoch

    def epoch_train(self, x, y):

        # Shuffle dataset in each epoch to avoid order bias
        # And execute a single epoch 
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
            alpha = min ( self.k / self.tau, 1 )
            self.eta = (1 - alpha) * self.eta_zero + alpha * self.eta_tau
            self.step(grad_b, grad_w, len(x_batch))
            self.k += 1

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

            # Bound the gradient norm (Pascanu et al)
            if self.max_norm > 0:
                norm = np.linalg.norm(grad_w[-l], axis=1)
                norm = np.where( norm < self.max_norm, 1, norm )
                grad_w[-l] = grad_w[-l] / norm[:,None]

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
