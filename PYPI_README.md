# NumCompute 

**NumCompute** is a lightweight, NumPy-inspired numerical computing library built for learning, experimentation, and basic machine learning workflows.

It provides simple and efficient implementations of common numerical operations such as statistics, preprocessing, optimization, metrics, and pipeline utilities.

---

# Overview

NumCompute is designed to simplify numerical computing in Python by offering easy-to-use functions similar to NumPy and Scikit-learn but in a lightweight, educational form.
It can be used in machine learning and deep learning module packages

---

# Key Features

- Basic statistical operations (mean, median, variance, etc.)
- Data preprocessing utilities (normalization, scaling)
- Sorting and searching algorithms
- Evaluation metrics for ML models
- Simple optimization tools
- Pipeline support for chaining operations

---

# Installation

Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ numcompute
```

# Example and Usage

Import the library and start using numerical functions easily. Here, is the example of the package in usage

```bash
import numcompute as nc

data = [10, 20, 30, 40, 50]

# Statistical operations
print(nc.mean(data))
print(nc.median(data))
print(nc.variance(data))

# Preprocessing
print(nc.normalize(data))

# Sorting / Searching
print(nc.sort(data))
print(nc.binary_search(data, 30))
```

# Author

Sabyasachi Kumar