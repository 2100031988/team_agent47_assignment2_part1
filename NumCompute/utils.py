import numpy as np


# =========================
# DISTANCES
# =========================
def euclidean_distance(a, b):
    """
    Compute Euclidean distance between vectors or batches.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):
    """
    L1 distance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sum(np.abs(a - b), axis=-1)


# =========================
# ACTIVATIONS
# =========================
def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def relu(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)


def softmax(x):
    """
    Numerically stable softmax.
    """
    x = np.asarray(x, dtype=float)

    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =========================
# LOG-SUM-EXP (STABILITY)
# =========================
def logsumexp(x, axis=None):
    """
    Stable log-sum-exp trick.
    """
    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


# =========================
# TOP-K HELPER
# =========================
def top_k(x, k, largest=True):
    """
    Return top-k values and indices.
    """
    x = np.asarray(x)

    if largest:
        idx = np.argpartition(x, -k)[-k:]
    else:
        idx = np.argpartition(x, k)[:k]

    vals = x[idx]

    order = np.argsort(-vals if largest else vals)
    return vals[order], idx[order]


# =========================
# BATCHING UTILITIES
# =========================
def batch_iterator(X, batch_size=32, shuffle=True):
    """
    Yield batches from dataset.
    """
    X = np.asarray(X)
    n = len(X)

    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx]