import numpy as np


# =========================
# Stable Sort
# =========================
def stable_sort(X, axis=-1):
    """
    Stable sort wrapper using NumPy.

    Parameters:
        X (array-like): input array
        axis (int): axis to sort along

    Returns:
        np.ndarray: sorted array
    """
    X = np.asarray(X)
    return np.sort(X, axis=axis, kind="stable")


# =========================
# Multi-key Sort
# =========================
def multi_key_sort(X, keys):
    """
    Sort rows of a 2D array using multiple column keys.

    Parameters:
        X (ndarray): shape (n_samples, n_features)
        keys (list[int]): column priority order

    Returns:
        np.ndarray: sorted array
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("multi_key_sort expects a 2D array")

    # reverse keys because lexsort uses last key as primary
    sort_indices = np.lexsort([X[:, k] for k in reversed(keys)])
    return X[sort_indices]


# =========================
# Top-K (Partial Sort)
# =========================
def topk(values, k, largest=True, return_indices=True):
    """
    Return top-k elements using np.argpartition.

    Parameters:
        values (array-like)
        k (int)
        largest (bool): True = largest k elements
        return_indices (bool)

    Returns:
        values or (values, indices)
    """
    values = np.asarray(values)

    if k <= 0:
        raise ValueError("k must be > 0")

    k = min(k, len(values))

    if largest:
        idx = np.argpartition(values, -k)[-k:]
    else:
        idx = np.argpartition(values, k)[:k]

    top_vals = values[idx]

    # sort final output
    order = np.argsort(-top_vals if largest else top_vals)
    idx = idx[order]
    top_vals = top_vals[order]

    return (top_vals, idx) if return_indices else top_vals


# =========================
# Binary Search
# =========================
def binary_search(sorted_array, x):
    """
    Binary search using NumPy searchsorted.

    Parameters:
        sorted_array (array-like): must be sorted
        x: value to search

    Returns:
        (index, found)
    """
    arr = np.asarray(sorted_array)

    idx = np.searchsorted(arr, x)
    found = (idx < len(arr)) and (arr[idx] == x)

    return idx, bool(found)


# =========================
# Quickselect (Educational)
# =========================
def quickselect(arr, k):
    """
    Find k-th smallest element using partition method.

    Parameters:
        arr (array-like)
        k (int)

    Returns:
        value at kth position
    """
    arr = np.asarray(arr).copy()

    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")

    return np.partition(arr, k)[k]