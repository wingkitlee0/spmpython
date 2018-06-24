"""
Spectral methods in MATLAB. Lloyd
Program CHEB
"""

# CHEB compute D = differentiation matrix, x = Chebyshev grid

import numpy as np
from scipy.sparse import diags
from scipy.linalg import toeplitz

def cheb(N):
    """
    # CHEB compute D = differentiation matrix, x = Chebyshev grid
    """
    if (N == 0):
        D = 0
        x = 1
    x = np.cos(np.pi*np.arange(0,N+1)/N)
    c = np.hstack([2, np.ones(N-1), 2])*(-1)**np.arange(0,N+1)
    X = np.tile(x,(N+1,1))
    dX = X.T - X
    D = (c[:,np.newaxis]*(1.0/c)[np.newaxis,:])/(dX+(np.identity(N+1)))       # off-diagonal entries
    D = D - np.diag(D.sum(axis=1))              # diagonal entries
    return D, x
    