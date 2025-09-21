'Optimization algorithms and functions'

# Standard libraries
import os
import sys
import math
import random
import datetime

# Numerical computing
import numpy as np

from costs import *


def compute_gradient(y, tx, w):
    """Computes the gradient at w in MSE case

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    #sample size
    N=len(y)

    #compute error
    e=y - tx @ w

    #compute gradient
    grad=(-1/N)* (tx.T @ e)

    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: w optimal through GD
        loss: loss function value evaluated at w optimal
        
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # computing gradient and loss
       
        grad=compute_gradient(y, tx, w)

        loss=compute_loss(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad

    return w, loss

def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-z))


def compute_logistic_gradient(y, tx, w):
    """Computes the gradient at w in logistic loss function case

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(d, ). The vector of model parameters.

    Returns:
        An array of shape (d, ) (same shape as w), containing the gradient of the loss at w.
    """
    #sample size
    N=len(y)

    #compute logistic function
    z = tx @ w
    sigma = sigmoid(z)
    
    #compute gradient
    grad=(1/N)* tx.T @ (sigma - y)

    return grad

def logistic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: w optimal through GD
        loss: loss function value evaluated at w optimal
        
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # computing gradient and loss
        grad=compute_logistic_gradient(y, tx, w)
        loss=compute_logistic_loss(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad

    return w, loss




