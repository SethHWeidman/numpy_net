import numpy as np

from helpers import sigmoid

def learn_complex_net(X, y):
    # This function learns the weights for a neural net that calculates one set
    # of "intermediate inputs" and then uses those inputs to make a prediction.

    # The multiplications are set up so that in each iteration, the weights are
    # indeed updated in the correct direction. To understand why, follow the
    # arguments here: http://sethweidman.com/neural_net_post_2
    np.random.seed(2)
    V = np.random.randn(3, 4)
    W = np.random.randn(4, 1)
    for j in range(50000):
        A = np.dot(X,V)
        B = _sigmoid(A)
        C = np.dot(B,W)
        P = _sigmoid(C)
        L = 0.5 * (y - P) ** 2
        dLdP = -1.0 * (y-P)
        dPdC = _sigmoid(C) * (1-_sigmoid(C))
        dLdC = dLdP * dPdC
        dCdW = B.T
        dLdW = np.dot(dCdW, dLdC)
        dCdB = W.T
        dLdB = np.dot(dLdC, dCdB)
        dBdA = _sigmoid(A) * (1-_sigmoid(A))
        dLdA = dLdB * dBdA
        dAdV = X.T
        dLdV = np.dot(dAdV, dLdA)
        W -= dLdW
        V -= dLdV
    return V, W

def predict_with_complex_net(X, V, W):
    # This function takes in the weights of a neural net that has been trained
    # using the 'learn_simple_net' function above and returns the prediction
    # that the function makes using these weights.
    A = np.dot(X, V)
    B = _sigmoid(A)
    C = np.dot(B, W)
    P = _sigmoid(C)
    return P

if __name__ == "__main__":
    # initialize X as a 1 x 3 matrix
    X = np.array([[0,0,1]])

    # initialize y as a 1 x 1 matrix
    y = np.array([[1]])

    # learn the weights
    V, W = learn_complex_net(X, y)

    # predict 'X' using the computed weights
    prediction = predict_with_complex_net(X, V, W)

    print "Prediction: %s" % prediction
    print "y: %s" % y
