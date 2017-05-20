'''Class objects for neural net'''
import numpy as np
from activation_functions import sigmoid
from python_custom.misc import list_index_wraparound
from helpers import set_learning_rate

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forwardpass(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
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
    def __init__(self, n_in, n_out, 
                 activation_function, 
                 random_seed, 
                 learning_rates=[]):
        self.random_seed=random_seed
        np.random.seed(seed=self.random_seed)
        self.W = np.random.normal(loc=0.0, scale=1, size=(n_in, n_out))
        self.activation_function = activation_function
        self.learning_rates = learning_rates
        self.iteration = 0
        
    def fprop(self, layer_input):
        self.layer_input = layer_input
        self.activation_input = np.dot(layer_input, self.W)
        return self.activation_function(self.activation_input, bprop=False)

    def bprop(self, layer_grad):
        dPdAi = self.activation_function(self.activation_input, bprop=True)
        dLdAi = layer_grad * dPdAi
        dAodAi = self.layer_input.T
        output_grad = np.dot(dLdAi, self.W.T)
        current_learning_rate = set_learning_rate(self.learning_rates, 
                                                  self.iteration)
        # if self.iteration % 100 == 0:
        #     print("The current learning rate is ", current_learning_rate)
        W_new = self.W - current_learning_rate * np.dot(dAodAi, dLdAi)
        self.W = W_new
        self.iteration += 1
        return output_grad
    


