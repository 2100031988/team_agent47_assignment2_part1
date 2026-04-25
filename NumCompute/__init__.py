"""
NumCompute: A lightweight NumPy-based machine learning framework.

This package provides:
- Data IO utilities
- Preprocessing tools (scalers, encoders, imputers)
- Statistical functions
- Sorting and search algorithms
- Ranking and percentile tools
- ML evaluation metrics
- Numerical optimization (gradients, Jacobians)
- Pipeline abstraction system
"""

# Core modules
from .io import load_csv

from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer
)

from .stats import (
    mean,
    median,
    std,
    min,
    max,
    quantile,
    histogram
)

from .sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    binary_search,
    quickselect
)

from .rank import rank, percentile

from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    mse,
    roc_curve,
    auc
)

from .optim import grad, jacobian

from .pipeline import Pipeline, FeatureUnion, Transformer, Estimator

from numcompute import StandardScaler, Pipeline
__all__ = [
    "load_csv",

    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",

    "mean",
    "median",
    "std",
    "min",
    "max",
    "quantile",
    "histogram",

    "stable_sort",
    "multi_key_sort",
    "topk",
    "binary_search",
    "quickselect",

    "rank",
    "percentile",

    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",

    "grad",
    "jacobian",

    "Pipeline",
    "FeatureUnion",
    "Transformer",
    "Estimator"
]