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
from optimizations import *
from costs import *



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






