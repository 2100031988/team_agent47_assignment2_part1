import numpy as np
import time

from numcompute.stats import mean
from numcompute.sort_search import topk


# =========================
# HELPER: TIMER
# =========================
def time_function(func, *args, repeat=5):
    times = []

    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)  # best time


# =========================
# LOOP vs VECTORIZED (MEAN)
# =========================
def loop_mean(X):
    total = 0.0
    count = 0
    for x in X:
        total += x
        count += 1
    return total / count


def benchmark_mean():
    X = np.random.rand(1_000_000)

    t_loop = time_function(loop_mean, X)
    t_vec = time_function(mean, X)

    return {
        "task": "Mean",
        "loop_time": t_loop,
        "vectorized_time": t_vec,
        "speedup": t_loop / t_vec
    }


# =========================
# LOOP vs VECTORIZED (TOP-K)
# =========================
def loop_topk(X, k):
    return sorted(X, reverse=True)[:k]


def benchmark_topk():
    X = np.random.rand(1_000_000)
    k = 10

    t_loop = time_function(loop_topk, X, k)
    t_vec = time_function(topk, X, k)

    return {
        "task": "Top-K",
        "loop_time": t_loop,
        "vectorized_time": t_vec,
        "speedup": t_loop / t_vec
    }


# =========================
# RUN ALL BENCHMARKS
# =========================
def run_benchmarks():
    results = []

    results.append(benchmark_mean())
    results.append(benchmark_topk())

    print("\n=== Benchmark Results ===")
    print(f"{'Task':<10} {'Loop (s)':<12} {'Vectorized (s)':<18} {'Speedup':<10}")

    for r in results:
        print(f"{r['task']:<10} {r['loop_time']:<12.6f} {r['vectorized_time']:<18.6f} {r['speedup']:<10.2f}")

    return results


if __name__ == "__main__":
    run_benchmarks()