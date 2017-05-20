import numpy as np
from sklearn.datasets import fetch_mldata

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

def numpy_round(x, digits):
    return float(np.round(x, digits))