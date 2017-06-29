'''Class objects for neural net'''
import math
import numpy as np

from activation_functions import sigmoid
from python_custom.misc import list_index_wraparound
from helpers import set_learning_rate
from scipy.stats import truncnorm

class NeuralNetwork(object):
    """Class for a neural network that can learn arbitrarily complicated
    relationships between input and output.
    """
    def __init__(self, layers, random_seed,
                 learning_rate_layer_decay=1,
                 learning_rate=1.0,
                 dropout=1.0,
                 drop_connect=1.0):
        """The function that initializes the neural network.

        Parameters
        ----------
        layers : list
            a list of the layers that the input will be passed through to
            generate the output
        random_seed : int
            the random seed to use when initializing the weights in each layer
        learning_rate : float
            the learning rate that will be used at the first (closest to input)
            layer of the network
        learning_rate_layer_decay : float
            the amount that the learning rate will decay by as we move forward
            in the layers
        dropout : float
            the proportion of neurons to keep in each layer if applying dropout
        drop_connect : float
            the proportion of weights to keep in each layer if applying drop
            connect
        """

        self.layers = layers
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.learning_rate_layer_decay = learning_rate_layer_decay
        self.dropout = dropout
        self.drop_connect = drop_connect
        self.iteration = 0
        for i, layer in enumerate(self.layers):
            setattr(layer, 'learning_rate',
                    self.learning_rate/(learning_rate_layer_decay ** i))
            setattr(layer, 'momentum', self.momentum)
            setattr(layer, 'random_seed', self.random_seed+i)
            setattr(layer, 'learning_rate', self.learning_rate)
            layer.initialize_weights()
            setattr(layer, 'drop_connect', self.drop_connect)

    def forwardpass(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for i, layer in enumerate(self.layers):
            np.random.seed(seed=self.random_seed+i*self.iteration)
            if self.dropout:
                zero_indices = np.random.choice(range(layer.n_in),
                                                size=int(layer.n_in * (1 - self.dropout)),
                                                replace=False)
                X_next[:, zero_indices] = 0.0
            X_next = layer.fprop(X_next)
        prediction = X_next
        return prediction

    def loss(self, prediction, Y):
        """ Calculate error on the given data. """
        loss = 0.5 * (Y - prediction) ** 2
        return -1.0 * (Y - prediction)

    def backpropogate(self, loss):
        """ Calculate an output Y for the given input X. """
        loss_next = loss
        for layer in reversed(self.layers):
            loss_next = layer.bprop(loss_next)
        return loss


class Layer(object):
    def _setup(self, input_shape, rng):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()


class Linear(Layer):

    random_seed = None
    learning_rate = None
    momentum = None

    def __init__(self, n_in, n_out,
             activation_function):
        self.n_in = n_in
        self.n_out = n_out
        self.iteration = 0
        self.activation_function = activation_function
        self.velocity = None

        if weights_scale == "xavier":
            weights_scale = 1.0 / math.sqrt(n_in)
        if weights_shape=="normal":
            self.W = np.random.normal(loc=0.0, scale=weights_scale, size=(n_in, n_out))
        elif weights_shape=="truncated":
            self.W = truncnorm.rvs(-1.0 * truncation, truncation,
                                   scale=weights_scale, size=(n_in, n_out),
                                   random_state=self.random_seed)
        else:
            return "Invalid weights shape"

    def initialize_weights(self):
        np.random.seed(seed=self.random_seed)
        self.W = np.random.normal(size=(self.n_in, self.n_out))

    def apply_drop_connect_weights(weights, drop_connect):
        new_weights = weights.copy()
        num_weights = new_weights.shape[0] * new_weights.shape[1]
        reshaped_weights = np.reshape(new_weights, (num_weights, 1))
        zero_indices = np.random.choice(range(num_weights),
                                        size=int(num_weights * (1 - drop_connect)),
                                        replace=False)
        reshaped_weights[zero_indices, :] = 0.0
        drop_connected_weights = np.reshape(reshaped_weights, new_weights.shape)

        return drop_connected_weights

    def fprop(self, layer_input):
        self.layer_input = layer_input
        if self.drop_connect:
            drop_connected_weights = apply_drop_connect_weights(self.W,
                                                                self.drop_connect)
            self.activation_input = np.dot(layer_input,
                                           drop_connected_weights)
        else:
            self.activation_input = np.dot(layer_input, self.W)
        self.iteration += 1
        return self.activation_function(self.activation_input, bprop=False)

    def bprop(self, layer_grad):
        dOutdActivationInput = self.activation_function(self.activation_input,
                                                        bprop=True)
        dLayerInputdActivationInput = layer_gradient * dOutdActivationInput
        dActivationOutputdActivationInput = self.layer_input.T
        output_grad = np.dot(dLayerInputdActivationInput, self.W.T)

        # Update velocity
        weight_update_current = np.dot(dActivationOutputdActivationInput,
                                       dLayerInputdActivationInput)
        self.velocity = np.add(self.momentum * self.velocity,
                               self.learning_rate * weight_update_current)
        self.W = self.W - self.velocity
        self.iteration += 1
        return output_grad
