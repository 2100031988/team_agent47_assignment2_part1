import numpy as np


# =========================
# RANKING (WITH TIES)
# =========================
def rank(data, method="average"):
    """
    Compute ranks with tie handling.

    Parameters:
        data (array-like)
        method: 'average' | 'dense' | 'ordinal'

    Returns:
        np.ndarray of ranks
    """
    x = np.asarray(data)

    # argsort twice gives stable ranking order
    order = np.argsort(x, kind="mergesort")

    ranks = np.empty(len(x), dtype=float)

    if method == "ordinal":
        ranks[order] = np.arange(1, len(x) + 1)

    elif method == "average":
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2
            i = j

    elif method == "dense":
        sorted_x = x[order]
        rank_val = 1
        i = 0

        while i < len(x):
            j = i
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1

            ranks[order[i:j]] = rank_val
            rank_val += 1
            i = j

    else:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'")

    return ranks


# =========================
# PERCENTILE FUNCTION
# =========================
def percentile(data, q, interpolation="linear"):
    """
    Compute percentile with interpolation methods.

    Parameters:
        data (array-like)
        q (float): 0–100
        interpolation: 'linear' | 'lower' | 'higher' | 'midpoint'

    Returns:
        float
    """
    x = np.asarray(data)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan

    x = np.sort(x)

    if q < 0 or q > 100:
        raise ValueError("q must be between 0 and 100")

    pos = (q / 100) * (len(x) - 1)

    if interpolation == "lower":
        return x[int(np.floor(pos))]

    elif interpolation == "higher":
        return x[int(np.ceil(pos))]

    elif interpolation == "midpoint":
        lo = x[int(np.floor(pos))]
        hi = x[int(np.ceil(pos))]
        return (lo + hi) / 2

    elif interpolation == "linear":
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))

        if lo == hi:
            return x[lo]

        return x[lo] + (pos - lo) * (x[hi] - x[lo])

    else:
        raise ValueError("Invalid interpolation method")