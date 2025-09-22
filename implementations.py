'Machine Learning methods implementations'

# Standard libraries
import os
import sys
import math
import random
import datetime

# Numerical computing
import numpy as np

#importing optimization functions
from optimization import *
from costs import *
from helpers import batch_iter


def compute_mse(y, tx, w):
    """Calculate the MSE loss.

    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    N = y.shape[0]
    return (1 / (2 * N)) * (e.T @ e)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - tx @ w
    return (-1 / N) * tx.T @ e


####### Gradient Descent

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        loss: a scalar denoting the loss value (scalar) for the optimal model parameters.
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD.
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        # Update w
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


####### SGD

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent (SGD) algorithm.

    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        initial_w: numpy array of shape=(D, ).
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.
    Returns:
        w: numpy array of shape=(D, ). The optimal model parameters.
        loss: a scalar denoting the loss value (scalar) for the optimal model parameters.
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        y_i, x_i = next(batch_iter(y, tx, batch_size=1))
        grad = compute_gradient(y_i, x_i, w)
        # Update w
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    'Ridge regression using normal equations'
    'INPUTS:
    '        - y = numpy  array of shape (N,) containing train outputs
    '        - tx = numpy array of shape (N, d) containing train inputs
    '        - lambda_ = ridge regression parameter
    'OUTPUTS:
    '        -w = numpy arraing containing the solution paramters
    '        -loss= the loss function value corresponding to the solution parameters, WITHOUT penalizing term
    # sample size
    N= len(y)

    #optimal parameters vector
    lambda_1= lambda_ * 2 * N
    w= (np.linalg.inv(tx.T @ tx + lambda_1 @ np.eye(tx.shape[1])))@ tx.T @ y

    #loss L(w) without penalizing term
    loss = compute_loss(y, tx, w)

    return w, loss




def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y ∈ {0, 1})

    INPUTS:
                - y = numpy  array of shape (N,) containing train outputs (0, 1)
                - tx = numpy array of shape (N, d) containing train inputs
                - initial_w = initial weight vector of paramters
                - max_iters= max number of iterations allowed in gradient descendent algorithm
                - gamma = step-size
    OUTPUTS:
                - w = numpy arraing containing the solution paramters
                - loss= the loss function value corresponding to the solution parameters
    """

    w, loss = logistic_gradient_descent(y, tx, initial_w, max_iters, gamma) #w optimal through logistic gradient descendent

    return w, loss






