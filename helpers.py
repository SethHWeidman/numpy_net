import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from python_custom.misc import (numpy_round, 
                                list_index_wraparound)

def get_mnist_X_Y():
    mnist = fetch_mldata('MNIST original')
    data = mnist.data
    X = (data - data.min()) * 1.0 / (data.max() - data.min())
    target = mnist.target
    Y = np.zeros((len(target), 10))
    for i in range(len(target)):
        Y[i][int(target[i])] = 1 
    return X, Y


def learn_neural_net(net, x, y):
    pred = net.forwardpass(x)
    loss = net.loss(pred, y)
    net.backpropogate(loss)
    return pred


def momentum_range(start_learning_rate, end_learning_rate, iterations):
    update = (end_learning_rate / start_learning_rate) ** (1.0 / iterations)
    learning_rates = [start_learning_rate * (update ** x) for x in range(iterations)]
    return learning_rates


def set_learning_rate(learning_rates, iteration):
    if len(learning_rates) == 0:
        current_learning_rate = 1
    else: 
        current_learning_rate = list_index_wraparound(learning_rates, 
                                                      iteration)
        # print("In this iteration of backprop, the learning rate is ", 
        #      current_learning_rate)
    return current_learning_rate


def _plot_learning_rate_decay(start_learning_rate, end_learning_rate, iterations): 
    plt.plot(range(iterations), momentum_range(start_learning_rate, end_learning_rate, iterations))
    plt.xlabel('Training iteration')
    plt.ylabel('Learning rate')
    plt.title('Learning rate decay illustration')