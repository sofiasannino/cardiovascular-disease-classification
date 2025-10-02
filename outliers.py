import numpy as np
import math


"""
def quicksort(x, p, r) :
    
    Inputs : x numnpy array
             p initial index
             r final index
    Outputs : x sorted

    
    if p < r : 
        q = partition(x, p, r)
        quicksort(x, p, q-1) # recursively sort the low side
        quicksort(x, q+1, r)    # recursively sort the high side
    return x
    


def partition(x, p, r) :
    pivot = x[r]
    i = p-1 #highest index into the low side
    for j in range(p, r-1) : #process each element other than the pivot 
        if x[j] <= pivot : 
            i = i+1
            temp = x[i]
            x[i] = x[j]
            x[j] = temp
    temp = x[i+1]
    x[i+1]=x[r]
    x[r] = temp
    return i + 1
"""

import numpy as np

def iqr(x, lower_quantile=0.05, upper_quantile=0.95):
    """
    Univariate outlier detection using adaptive IQR based on given percentiles.
    
    Args:
        x : numpy array of values
        lower_quantile : float, lower percentile (default 5%)
        upper_quantile : float, upper percentile (default 95%)
        
    Returns:
        numpy array of indices of outliers
    """
    x = np.array(x, dtype=float)
    x_valid = x[~np.isnan(x)]
    if len(x_valid) == 0:
        return np.array([], dtype=int)

    Q1 = np.percentile(x_valid, lower_quantile*100)
    Q3 = np.percentile(x_valid, upper_quantile*100)
    IQR = Q3 - Q1

    mask = ((x < Q1 - 1.5*IQR) | (x > Q3 + 1.5*IQR)) & (~np.isnan(x))
    return np.where(mask)[0]


def fast_mcd(x, h, m,  max_iters, tol,  seed) :
    """
    Args : 
        x numpy array containing the data
        h subsample length
        m number of random subsamples starters
        max_iters maximum number of iterations allowed per random subsample
        tol tolerance for stability of the covariance determinant
        seed randomization seed
        
    Outputs : 
        mu robust means numpy array
        Sigma robust covariance matrix numpy array
        d numpy array containing Mahalanobis distances >> large values of di indicate 
        outliers 

    """

   
    n, d = x.shape
    best_det = np.inf
    best_mu, best_Sigma, best_d = None, None, None

    rng = np.random.default_rng(seed)

    for k in range(m):
         
        #computing random subsample
        rand_subsample_indexes = rng.choice(n, size=h, replace = False) #generating h random indexes in the dataset
        subsample=x[rand_subsample_indexes, :]

        #loop parameters
        it = 0
        res = tol + 1

        #computing mean and covariance matrix estimates of the initial random subsample 
        mu = subsample.mean(axis=0)
        Sigma = np.cov(subsample, rowvar=False, bias=False)
        detk = np.linalg.det(Sigma)
        d= np.zeros(n,)

        while it < max_iters and res > tol :

            # compute Mahalanobi distances for each element of the dataset
            diff = x-mu
            d = np.sum(diff.T * np.linalg.solve(Sigma, diff.T), axis=0)



            #select new subsample of h elements with the smallest M. distance
            subsample_indexes = np.argsort(d)[:h]
            subsample = x[subsample_indexes]

            #computing mean and covariance matrix estimates of the subsample 
            mu = subsample.mean(axis=0)
            Sigma = np.cov(subsample, rowvar=False, bias=False)
            detk1 = np.linalg.det(Sigma)

            #checking convergence and updating
            res = np.abs(detk- detk1)/(np.abs(detk) + 1e-12)
            detk = detk1
            it = it + 1

        # keep best solution
        if detk < best_det and detk > 0:
            best_det = detk
            best_mu, best_Sigma, best_d = mu, Sigma, d

    return best_mu, best_Sigma, best_d









    




