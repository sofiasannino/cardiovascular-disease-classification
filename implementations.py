'Machine Learning methods implementations'

# Standard libraries
import os
import sys
import math
import random
import datetime

# Numerical computing
import numpy as np
from helpers import batch_iter



# ********************************************************************************
### COMPUTATION OF LOSSES ###
# ********************************************************************************
def compute_logistic_loss(y, tx, w, lambda_ = 0):
    """
        Calculate logistic loss when y is in {0, 1}
    Args: 
        - y = numpy array of shape (N, ) containing training outputs
        - tx = numpy array of shape (N, d) containing training inputs
        - w = numpy array of shape (d, ) containing parameters
        - lambda_ : regularization parameter
    Returns: 
        - loss = logistic loss value at w
    """
    #sample size
    N = len(y)

    #compute loss
    z = tx @ w
    loss = (1/N)*(np.sum(- y * z + np.log( 1 + np.exp(z)))) + lambda_ *(( np.linalg.norm(w) )**2)

    return loss


def compute_mse_loss(y, tx, w):
    """Calculate the MSE loss.

    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the MSE loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    N = y.shape[0]
    return (1 / (2 * N)) * (e.T @ e)





# *******************************************************************************************
### GRADIENTS COMPUTATION ###
# *******************************************************************************************
def compute_gradient(y, tx, w):
    """Computes the gradient at w of MSE loss.

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

def sigmoid(z):
    """Numerically stable sigmoid (avoiding overflow)."""
    result = np.empty_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    result[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
    return result


def compute_logistic_gradient(y, tx, w, lambda_):
    """Computes the gradient of logistic loss function at w. L2 regularization is used.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d, ). The vector of model parameters.
        lambda_ : regularization parameter

    Returns:
        An array of shape (d, ) (same shape as w), containing the gradient of the L2-regularized loss at w.
    """
    #sample size
    N=len(y)

    #compute logistic function
    z = tx @ w
    sigma = sigmoid(z)
    
    #compute gradient
    grad=(1/N)* tx.T @ (sigma - y) + 2 * lambda_ * w

    return grad



# ***********************************************************************
### OPTIMIZATION ALGORITHMS ###
# ***********************************************************************
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y ∈ {0, 1})

    INPUTS:
                - y = numpy  array of shape (N,) containing train outputs (0, 1)
                - tx = numpy array of shape (N, d) containing train inputs
                - initial_w = initial weight vector of parameters
                - max_iters= max number of iterations allowed in gradient descendent algorithm
                - gamma = step-size
    OUTPUTS:
                - w = numpy array containing the solution parameters
                - loss= the loss function value corresponding to the solution parameters
    """
    w = initial_w
    lambda_ = 0
    
    for n_iter in range(max_iters):
        # computing gradient and loss
        grad=compute_logistic_gradient(y, tx, w, lambda_)
        # update w by gradient
        w = w - gamma * grad

    loss=compute_logistic_loss(y, tx, w,lambda_) #compute optimal loss

    return w, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm applied to linear regression with MSE loss function.

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

    loss = compute_mse_loss(y, tx, w)
    return w, loss



def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent (SGD) algorithm for linear regression with MSE loss function.

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
    
    for n_iter in range(max_iters):
        y_i, x_i = next(batch_iter(y, tx, batch_size=1))
        grad = compute_gradient(y_i, x_i, w)
        # updating w
        w = w - gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations'
    INPUTS:
            - y = numpy  array of shape (N,) containing train outputs
            - tx = numpy array of shape (N, d) containing train inputs
            - lambda_ = ridge regression parameter
    OUTPUTS:
            -w = numpy array containing the solution parameter
            -loss= the loss function value corresponding to the solution parameters, WITHOUT penalizing term
    """
    #sample size
    N= len(y)

    #optimal parameters vector
    lambda_1= lambda_ * 2 * N
    A=tx.T @ tx + lambda_1 * np.eye(tx.shape[1])
    b=tx.T @ y
    w= np.linalg.solve(A, b)

    #loss L(w) without penalizing term
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares(y,tx):
    
    """
    Linear regression using least squares
    
    Args:
    
        tx: numpy array of shape=(N,D)
        y: numpy array of shape=(N, )
        
    Returns:
    
        w: numpy array of shape=(N, ). The optimal model parameters.
        loss: a scalar denoting the loss value (scalar) for the optimal model parameters. 
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y ∈ {0, 1}).

    Args :
        y : numpy array of shape = (N, ).
        tx : numpy array of shape = (N, D).
        lambda_ : a scalar denoting the regularization parameter.
        initial_w : numpy array of shape = (D, ).
        max_iters : a scalar denoting the total number of iterations of GD.
        gamma : a scalar denoting the stepsize.
    Returns :
        w : numpy array of shape = (D, ). The optimal model parameters.
        loss : a scalar denoting the loss value for the optimal model parameters, without considering the penalized term.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
        
    loss = compute_logistic_loss(y, tx, w, lambda_=0)

    return w, loss



# ***********************************************************
### ADDITIONAL OPTIMIZATION ALGORITHM ###
#************************************************************
def reg_logistic_regression_adam(y, tx, lambda_, initial_w, max_iters,beta_1, beta_2, gamma, mini_batch_size):
    """L2-Regularized logistic regression using mini-batch Adam algorithm (y ∈ {0, 1}).

    Args :
        y : numpy array of shape = (N, ).
        tx : numpy array of shape = (N, D).
        lambda_ : a scalar denoting the regularization parameter.
        initial_w : numpy array of shape = (D, ).
        max_iters : a scalar denoting the total number of iterations of Adam SGD.
        beta_1 : a scalar ∈ [0, 1] that defines the momentum parameter.
        beta_2 : a scalar ∈ [0, 1] that defines the average of past squared gradients parameter.
        gamma : a scalar denoting the stepsize.
        mini_batch_size : size of the minibatch randomly chosen to compute gradient 
    Returns :
        w : numpy array of shape = (D, ). The optimal model parameters.
        loss : a scalar denoting the logistic loss value for the optimal model parameters, without considering the penalization term.
    """
    # Initial parameters
    w = initial_w
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    it = 0 #iterations
    eps = 1e-8

    # Applying Adam
    for n_iter in range(max_iters):
        it = it + 1
        y_i, x_i = next(batch_iter(y, tx, mini_batch_size))
        gradient = compute_logistic_gradient(y_i, x_i, w, lambda_)

        # Updating momentum and second raw momentum
        m = beta_1 * m + (1- beta_1) * gradient #momentum 
        v = beta_2 * v + (1- beta_2) * (gradient ** 2) #squared gradients average
        
        
        # Unbiased first and second momentum
        m_hat = m /(1 - beta_1 ** it)
        v_hat= v / (1 - beta_2 ** it)
        
        w = w - (gamma/(np.sqrt(v_hat)+ eps)) * m_hat

    # Computing optimal loss, without penalization term 
    loss = compute_logistic_loss(y, tx, w, lambda_=0)

    return w, loss












