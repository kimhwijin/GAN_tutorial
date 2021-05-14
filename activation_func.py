import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def ELU(x,alpha):
    return x if x > 0 else alpha * (np.exp(x) - 1)

def LeakyReLU(x,alpha):
    '''
    params :
    x : np data
    alpha : alpha > 0 number
    '''
    return x if x > 0 else alpha * x
