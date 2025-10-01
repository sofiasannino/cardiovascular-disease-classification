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
    Outlier detection using adaptive IQR based on given percentiles.
    
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






    




