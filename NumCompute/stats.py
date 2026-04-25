import numpy as np


def mean(X, axis=None):
    """
    Compute mean (NaN-safe).
    """
    X = np.asarray(X)
    return np.nanmean(X, axis=axis)


def median(X, axis=None):
    """
    Compute median (NaN-safe).
    """
    X = np.asarray(X)
    return np.nanmedian(X, axis=axis)


def std(X, axis=None, ddof=0):
    """
    Compute standard deviation (NaN-safe).
    """
    X = np.asarray(X)
    return np.nanstd(X, axis=axis, ddof=ddof)


def min(X, axis=None):
    """
    Compute minimum (NaN-safe).
    """
    X = np.asarray(X)
    return np.nanmin(X, axis=axis)


def max(X, axis=None):
    """
    Compute maximum (NaN-safe).
    """
    X = np.asarray(X)
    return np.nanmax(X, axis=axis)


def quantile(X, q, axis=None):
    """
    Compute quantiles (NaN-safe).

    Parameters:
        q : float or array-like (0 to 1)
    """
    X = np.asarray(X)
    return np.nanquantile(X, q, axis=axis)


def histogram(X, bins=10, range=None):
    """
    Compute histogram.

    Returns:
        counts, bin_edges
    """
    X = np.asarray(X)
    X = X[~np.isnan(X)]  # remove NaNs

    counts, bin_edges = np.histogram(X, bins=bins, range=range)
    return counts, bin_edges

