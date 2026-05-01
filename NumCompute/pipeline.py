import numpy as np

# pipeline.py aims to combine multiple steps into a workflow and that includes collecting raw data then preprocessing it, then applying some transformations and finally fitting a model to the data
# where we do some transformations on the data to get desired output and then we can use the output to make predictions or evaluate the model performance.


# =========================
# BASE CLASSES (Protocol)
# =========================

class Transformer:                                  
    """
    Base Transformer interface.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Estimator:
    """
    Base Estimator interface (for models).
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError


# =========================
# PIPELINE
# =========================
class Pipeline:
    """
    Sequential pipeline for transformers and estimators.
    """

    def __init__(self, steps):
        """
        steps: list of (name, transformer/estimator)
        """
        self.steps = steps

    def fit(self, X, y=None):
        for _, step in self.steps:
            if hasattr(step, "fit"):
                if "y" in step.fit.__code__.co_varnames:
                    step.fit(X, y)
                else:
                    step.fit(X)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self

    def transform(self, X):
        for _, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# =========================
# FEATURE UNION (BONUS)
# =========================
class FeatureUnion:
    """
    Apply multiple transformers in parallel and concatenate outputs.
    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for _, t in self.transformers:
            t.fit(X, y)
        return self

    def transform(self, X):
        outputs = [t.transform(X) for _, t in self.transformers]
        return np.hstack(outputs)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)