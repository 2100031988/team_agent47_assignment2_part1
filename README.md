<h1 align="center">NumCompute: NumPy-Based ML Framework 🚀</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Enabled-orange?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/Tests-20+-brightgreen" alt="Tests">
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

## 🌟 Project Overview

**NumCompute** is a lightweight machine learning and numerical computing framework built using **NumPy**. It simulates core functionality of libraries like **scikit-learn**, while keeping the internals transparent and educational.

**Designed for:**
- 🎓 Learning ML algorithms from scratch
- 🔬 Academic assignments and evaluations
- 🧠 Building intuition for NumPy-based systems
- 🔧 Experimenting with end-to-end ML pipelines

---

## 🚀 Features

| Category | Features |
|----------|----------|
| 📥 **I/O** | CSV Loader with missing value handling |
| 🔄 **Preprocessing** | StandardScaler, LabelEncoder, Imputer |
| 🔍 **Algorithms** | Binary Search, Quickselect, Top-K |
| 📊 **Statistics** | Mean, Median, Std, Variance, Histogram |
| 🏆 **Ranking** | Tie-aware ranking system |
| 📈 **Metrics** | Accuracy, F1, MSE, ROC-AUC |
| ⚡ **Optimization** | Gradient & Jacobian estimation |
| 🔗 **Pipeline** | scikit-learn style ML pipeline |
| 🧪 **Testing** | 20+ unit tests with `pytest` |
| 📊 **Benchmarking** | Performance comparison tools |

---

## 📦 Installation

### Option 1: Install from TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ numcompute
```

### Option 2: Install from Source

```bash
git clone https://github.com/2100031988/NumCompute.git
cd NumCompute
pip install .
```

---

## ⚡ Usage

```python
import numcompute as nc

data = [10, 20, 30, 40, 50]

# 📊 Statistics
print("Mean:", nc.mean(data))
print("Median:", nc.median(data))
print("Std:", nc.std(data))

# 🔄 Preprocessing
print("Normalized:", nc.normalize(data))

# 🔍 Sorting & Searching
print("Sorted:", nc.sort(data))
print("Search 30:", nc.binary_search(data, 30))

# 📈 Metrics
print("MSE:", nc.mse([1, 2, 3], [1, 2, 4]))
```

### Pipeline Example

```python
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

## 📊 Benchmarking

NumCompute includes benchmarking tools to compare performance of custom implementations against optimized NumPy operations.

**Benchmarks include:**
- Sorting algorithm performance comparison
- Statistical function execution time
- Pipeline processing throughput

Run benchmarks:

```bash
python benchmark/run_benchmarks.py
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
└── LICENSE
```
---

## 🧪 Running Tests

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

## 🌐 TestPyPI Release

This project is published on TestPyPI:

👉 [https://test.pypi.org/project/numcompute/](https://test.pypi.org/project/numcompute/)

---

## 🎯 Project Goals

- [x] Understand ML algorithms from scratch
- [x] Build NumPy-like functionality manually
- [x] Apply software engineering principles (modular design, testing)
- [x] Create a reusable, scalable ML framework
- [x] Publish to TestPyPI

---

## 👨‍💻 Author

**Sabyasachi Kumar**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
