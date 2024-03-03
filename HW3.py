import numpy as np
import pandas as pd
from scipy.optimize import minimize

#we need to read in the data and split it by class? get params for each set? 
# or does Classifying 1 against 2 and 3 mean we classify as either class one, or of class 2/3? 

def sigmoid(x, w, threshold=0.5):
    p = 1 / (1 + np.exp(-x * w.transpose()))
    return np.where(p > threshold, 1, 0)

#TP,FP,TN,FN should be the number of each kind of error occuring 
def accuracy(TP, FP,TN,FN):
    return ((TP+TN)/(TP+TN+FP+FN))

def precision(TP,FP):
    return TP/(TP+FP)

def logistic_loss(w, X, y):
    sum = -(y * np.log(sigmoid(X, w)) + (1 - y) * np.log(1 - sigmoid(X, w)))
    return sum/len(X)


def logistic_loss_grad(w, X, y):
    return (X.transpose() * (sigmoid(X, w) - y)) / len(X)

#this gets us a new set of W's which form the w array we pass into the sigmoid 
# optimizedParams = minimize(logistic_loss, np.zeros(X.shape[1]), jac=logistic_loss_grad, args=(X, y), method="L-BFGS-B").x
