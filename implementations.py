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



### COMPUTATION OF LOSSES ###
def compute_logistic_loss(y, tx, w):
    """
        Calculate logistic loss when y is in {0, 1}
    Args: 
        - y = numpy array of shape (N, ) containing training outputs
        - tx = numpy array of shape (N, d) containing training inputs
        - w = numpy array of shape (d, ) containing parameters
    Returns: 
        - loss = logistic loss value at w
    """
    #sample size
    N = len(y)

    #compute loss
    z = tx @ w
    loss = (1/N)*(np.sum(- y * z + np.log( 1 + np.exp(z))))

    return loss



def compute_mse_loss(y, tx, w):
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



### GRADIENTS COMPUTATION ###
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


def compute_logistic_gradient(y, tx, w):
    """Computes the gradient at w in logistic loss function case

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
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





### OPTIMIZATION ALGORITHMS ###
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
    w = initial_w
    
    for n_iter in range(max_iters):
        # computing gradient and loss
        grad=compute_logistic_gradient(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad
    loss=compute_logistic_loss(y, tx, w) #compute optimal loss

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
    N = len(y)
    for n_iter in range(max_iters):
        y_i, x_i = next(batch_iter(y, tx, batch_size=1))
        grad = compute_gradient(y_i, x_i, w)
        # Update w
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
            -w = numpy arraing containing the solution paramters
            -loss= the loss function value corresponding to the solution parameters, WITHOUT penalizing term
    """
        # sample size
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
    
    ones = np.ones((tx.shape[0], 1)) 
    X = np.concatenate([ones, tx], axis=1)
    #X[np.isnan(X)] = 0
    #w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    w = np.linalg.solve(tx,y)
    
    loss = compute_mse(y,tx,w)
    
    return w, loss

def compute_reg_logistic_loss(y, tx, w,lambda_)
"""
        Calculate logistic loss when y is in {0, 1}
Args: 
        - y = numpy array of shape (N, ) containing training outputs
        - tx = numpy array of shape (N, d) containing training inputs
        - w = numpy array of shape (d, ) containing parameters
Returns: 
        - loss = logistic loss value at w
"""
    #sample size
    N = len(y)

    #compute loss
    z = tw @ w
    loss = (1/N)*(np.sum(- y * z + np.log( 1 + np.exp(z)) + lambda_*np.square(w)))

    return loss

def compute_reg_logistic_gradient(y, tx, w, lambda_):
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
    grad=(1/N)* tx.T @ (sigma - y) + 2*lambda_*w

    return grad

def reg_logistic_gradient_descent(y,tx,lambda_,initial_w,max_iters,gamma):
    """The Regularised Logistic Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda: regularisation term

    Returns:
        w: w optimal through GD
        loss: loss function value evaluated at w optimal
        
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # computing gradient and loss
        grad=compute_reg_logistic_gradient(y, tx, w,lambda_)
        loss=compute_reg_logistic_loss(y, tx, w, lambda_)
        
        # update w by gradient
        w = w - gamma * grad

    return w, loss

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """
    Regularised Logistic regression using gradient descent (y ∈ {0, 1})

    INPUTS:
                - y = numpy  array of shape (N,) containing train outputs (0, 1)
                - tx = numpy array of shape (N, d) containing train inputs
                - initial_w = initial weight vector of paramters
                - max_iters= max number of iterations allowed in gradient descendent algorithm
                - gamma = step-size
                - lambda: regularisation term
    OUTPUTS:
                - w = numpy arraing containing the solution paramters
                - loss= the loss function value corresponding to the solution parameters
    """
    w, loss = reg_logistic_gradient_descent(y, tx, lambda_,initial_w, max_iters, gamma) #w optimal through logistic gradient descendent
        
        
    return w, loss







