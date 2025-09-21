# -*- coding: utf-8 -*-
"""functions used to compute the loss."""

# Standard libraries
import os
import sys
import math
import random
import datetime

import numpy as np


def compute_loss(y, tx, w, MSE= True):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    #sample size
    N=len(y)
    
    #compute the error matrix
    e=y- tx @ w 

    # compute the MSE loss or MAE loss
    if MSE :
        Lw= (0.5*N)*(e @ e) #e.T @ e
    else :
        #Lw= (1/N) * (np.sum(np.sum ( np.abs(e.reshape(-1, 1)) , axis=1), axis=0))
        Lw=(1/N) * sum(abs(e))

return Lw


def compute_logistic_loss(y, tx, w)
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
    loss = (1/N)*(np.sum(- y * z + np.log( 1 + np.exp(z))))

return loss


    
