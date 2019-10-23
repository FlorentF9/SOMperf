"""
Neighborhood functions
"""

import numpy as np


def gaussian_neighborhood(radius=1.0):
    """Gaussian neighborhood kernel function.

    Parameters
    ----------
    radius : float (default = 1.0)
        standard deviation of the gaussian kernel.

    Returns
    -------
    neighborhood_fun : (d : int) => float in [0,1]
        neighborhood function.
    """
    def neighborhood_fun(d):
        return np.exp(- (d**2) / (radius**2))
    return neighborhood_fun


def window_neighborhood(radius=1.0):
    """Window neighborhood kernel function.

    Parameters
    ----------
    radius : float (default = 1.0)
        radius of the window.

    Returns
    -------
    neighborhood_fun : (d : int) => float in [0,1]
        neighborhood function.
    """
    def neighborhood_fun(d):
        return 1.0 if d <= radius else 0.0
    return neighborhood_fun
