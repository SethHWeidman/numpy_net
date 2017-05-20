import numpy as np

def sigmoid(x, bprop=False):
    if bprop:
        s = sigmoid(x)
        return s*(1-s)
    else:
        return 1.0/(1.0+np.exp(-x))
    
def linear(x, bprop=False):
    return x

def tanh(x, bprop=False):
    # TODO
    return None