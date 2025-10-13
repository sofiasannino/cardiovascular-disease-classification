'Useful methods for cross validation part'

# Standard libraries
import os
import sys
import math
import random
import datetime
import matplotlib.pyplot as plt

# Numerical computing
import numpy as np
from helpers import batch_iter

# importing machine learning algorithms
from implementations import *


def compute_auc(y_true, y_scores):
    """
    AUC calculation using Mann-Whitney statistics
    Inputs : 
            - y_true : numpy array containing the real {0, 1} values of the dataset
            - y_scores : numpy array containing our predictions
    Output : 
            AUC Area under the ROC curve 
    """
    order = np.argsort(y_scores)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    # rank positions 
    rank_positions = np.arange(1, len(y_true_sorted) + 1)
    rank_sum = np.sum(rank_positions[y_true_sorted == 1])

    # AUC using Mann–Whitney
    auc = (rank_sum - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return auc



def build_k_indices(N, k_fold, seed): 
    """build k indices for k-fold.

    Args:
        N:      num of samples
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = N
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation_reg_log(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma ):
    """Return the AUC, Accuracy, logistic loss (on train and test) for a fold corresponding to k_indices using 
        regularized logistic regression

    Args:
        y:          shape=(N,) numpy array output {0, 1}
        x:          shape=(N, d) numpy array input
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold 
        lambda_:    scalar, regularization parameter fo regularized logistic loss
        initial_w : numpy array (d, ), initial guess for the optimal paramters vector used in the regression
        max_iters : scalar, maximum number of iterations used in regression
        gamma : step size regression 

    Returns:
        - train and test logistic loss for k-th fold
        - AUC using compute_auc method for k-th fold
        - accuracy  for k-th fold 
    """

    #k-th subgroup in test, others in train
    test_mask = np.isin(np.arange(len(y)), k_indices[k, :])
    y_test = y[test_mask]
    x_test = x[test_mask]
    
    y_train=y[~test_mask]
    x_train=x[~test_mask]
    
    # regularized logistic regression 
    w_opt, loss_tr  = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma )
    
    # calculate the loss for test data
    loss_te = compute_logistic_loss(y_test, x_test, w_opt)

    # compute AUC 
    predictions = sigmoid (x_test @ w_opt)
    AUC = compute_auc(y_test, predictions)

    # compute accuracy
    y_pred = (predictions >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)

    return loss_tr, loss_te, AUC, accuracy

def cross_validation_adam(y, x, k_indices, k, lambda_, initial_w, max_iters, beta_1, beta_2,  gamma, mini_batch_size ):
    """Return the AUC, Accuracy, logistic loss (on train and test) for a fold corresponding to k_indices using 
        L2-Regularized logistic regression using mini-batch Adam algorithm

    Args:
        y:          shape=(N,) numpy array output {0, 1}
        x:          shape=(N, d) numpy array input
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold 
        lambda_:    scalar, regularization parameter fo regularized logistic loss
        initial_w : numpy array (d, ), initial guess for the optimal paramters vector used in the regression
        max_iters : scalar, maximum number of iterations used in regression
        beta_1 : a scalar ∈ [0, 1] that defines the momentum parameter.
        beta_2 : a scalar ∈ [0, 1] that defines the average of past squared gradients parameter.
        gamma : a scalar denoting the stepsize.
        mini_batch_size : size of the minibatch randomly chosen to compute gradient

    Returns:
        - train and test logistic loss for k-th fold
        - AUC using compute_auc method for k-th fold
        - accuracy  for k-th fold 
    """

    #k-th subgroup in test, others in train
    test_mask = np.isin(np.arange(len(y)), k_indices[k, :])
    y_test = y[test_mask]
    x_test = x[test_mask]
    
    y_train=y[~test_mask]
    x_train=x[~test_mask]
    
    # regularized logistic regression 
    w_opt, loss_tr  = reg_logistic_regression_adam(y, x, lambda_, initial_w, max_iters,beta_1, beta_2, gamma, mini_batch_size)
    
    # calculate the loss for test data
    loss_te = compute_logistic_loss(y_test, x_test, w_opt)

    # compute AUC 
    predictions = sigmoid (x_test @ w_opt)
    AUC = compute_auc(y_test, predictions)

    # compute accuracy
    y_pred = (predictions >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)

    return loss_tr, loss_te, AUC, accuracy


def cross_validation_visualization(lambds, logi_loss_tr, logi_loss_te):
    """visualization the curves of logi_loss_tr and logi_loss_te."""
    plt.semilogx(lambds, logi_loss_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, logi_loss_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("logistic regression loss")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation for hyperparameters tuning")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_lambda_reg_log_loss")