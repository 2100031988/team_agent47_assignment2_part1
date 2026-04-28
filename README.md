<h1 align="center">NumCompute: A Numpy Toolkit </h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Enabled-orange?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-benchmarking">Benchmarking</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-running-tests">Tests</a>
</p>

---

## Overview

**NumCompute** is a lightweight machine learning framework built using **NumPy**. It can function on core libraries like **scikit-learn** while keeping the internal function transparent and understandable for users.

**Designed usage:**
There are numerous usages of this library but some listed below : 

1. It is a perfect tool to evaluate assignment tools and validation.
2. We can learn about Machine Learning algorithms from scratch.
3. Lastly, the most important usage is for developing pipelines as we can experiment with many stages.


---

## Installation
We can install the project in your desktop through two steps mentioned below:

### Option 1: Install from TestPyPI
We can install it directly in your desktop by typing the below command in the **terminal** in visual studio code.

```bash
pip install -i https://test.pypi.org/simple/ numcompute
```

### Option 2: Install from Source 
We can also install it from my repository through github commands in the **terminal** in visual studio code.

```bash
git clone https://github.com/2100031988/NumCompute.git
cd NumCompute
pip install .
```

---

## Examples
Here the some of the example code chunks provided to help us better understand the code.

```python

import numcompute as nc
data = [10, 20, 30, 40, 50]

# <----- Statistics ----->
print("Mean:", nc.mean(data))
print("Median:", nc.median(data))
print("Std:", nc.std(data))


# <----- Preprocessing ----->
print("Normalized:", nc.normalize(data))


# <----- Sorting & Searching ----->
print("Sorted:", nc.sort(data))
print("Search 30:", nc.binary_search(data, 30))


# <----- Metrics ----->
print("MSE:", nc.mse([1, 2, 3], [1, 2, 4]))


# <----- Pipeline example ----->
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, LabelEncoder
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("encoder", LabelEncoder()),
])
pipeline.fit(X_train)
X_transformed = pipeline.transform(X_test)

```

---

## 📂 Project Structure

```bash
NumCompute/
│
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       ├── feature_request.md
│       ├── question.md
│       └── config.yml
│
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   └── utils.py
│
├── tests/
│   └── test_numcompute.py
│
├── demo/
│   └── quickstart.ipynb
│
├── benchmark/
│   └── benchmarking.py 
│
│
├── pyproject.toml
├── README.md
├── PYPI_README.md
└── .gitignore
└── LICENSE
```

---

## Running Tests

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=numcompute
```

---

## TestPyPI Release

This project is published on TestPyPI:

👉 [https://test.pypi.org/project/numcompute/](https://test.pypi.org/project/numcompute/)

---


## Contributors

**Sabyasachi Kumar**

---

## License

This project is licensed under the [MIT License](LICENSE).
