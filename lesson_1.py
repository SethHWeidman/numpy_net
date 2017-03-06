import numpy as np

from helpers import sigmoid

def learn_simple_net(X, y):
    # This function learns the weights for the simplest possible "neural net":
    # one with no hidden layer. This is conceptually equivalent to a logistic
    # regression.

    # The multiplications are set up so that in each iteration, the weights are
    # indeed updated in the correct direction. To understand why, follow the
    # argument here: http://sethweidman.com/neural_net_post
    np.random.seed(1)
    W = np.random.randn(3, 1)
    for i in range(500):
        A = np.dot(X, W)
        P = sigmoid(A)
        L = 0.5 * (y - P) ** 2
        if i % 50 == 0:
            print P
        dLdP = -1.0 * (y - P)
        dPdA = sigmoid(A) * (1.0 - _sigmoid(A))
        dLdA = dLdP * dPdA
        dAdW = X.T
        dLdW = np.dot(dAdW, dLdA)
        W -= dLdW
    return W

def predict_with_simple_net(X, W):
    # This function takes in the weights of a neural net that has been trained
    # using the 'learn_simple_net' function above and returns the prediction
    # that the function makes using these weights.
    A = np.dot(X, W)
    P = sigmoid(A)
    return P

if __name__ == "__main__":
    # initialize X as a 1 x 3 matrix
    X = np.array([[0,0,1]])

    # initialize y as a 1 x 1 matrix
    y = np.array([[1]])

    # learn the weights
    W = learn_simple_net(X, y)

    # predict 'X' using the computed weights
    prediction = predict_with_simple_net(X, W)

    print "Prediction: %s" % prediction
    print "y: %s" % y
