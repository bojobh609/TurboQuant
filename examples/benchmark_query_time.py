"""Honest query-time benchmark: TurboQuantIndex vs IVFTurboQuantIndex.

Measures latency and recall at multiple dataset sizes.
Run: python examples/benchmark_query_time.py
"""

import time
import numpy as np
from turboquant import TurboQuantIndex, IVFTurboQuantIndex


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def benchmark_brute_force(db, queries, k, num_bits):
    idx = TurboQuantIndex(dimension=db.shape[1], num_bits=num_bits, use_qjl=False)
    t0 = time.perf_counter()
    idx.add(db)
    add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sims, indices = idx.search(queries, k=k)
    query_time = time.perf_counter() - t0

    return add_time, query_time, indices


def benchmark_ivf(db, queries, k, num_bits, nlist, nprobe):
    idx = IVFTurboQuantIndex(
        dimension=db.shape[1],
        num_bits=num_bits,
        nlist=nlist,
        nprobe=nprobe,
        use_qjl=False,
    )
    t0 = time.perf_counter()
    idx.train(db)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx.add(db)
    add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sims, indices = idx.search(queries, k=k)
    query_time = time.perf_counter() - t0

    return train_time, add_time, query_time, indices


def compute_recall(pred_indices, gt_indices):
    recalls = []
    for i in range(len(pred_indices)):
        pred_set = set(int(x) for x in pred_indices[i] if x >= 0)
        gt_set = set(int(x) for x in gt_indices[i])
        if len(gt_set) > 0:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return np.mean(recalls)


def main():
    d = 384
    k = 10
    num_bits = 6
    num_queries = 100

    print("=" * 80)
    print(f"TurboQuant Query-Time Benchmark (d={d}, bits={num_bits}, k={k})")
    print("=" * 80)
    print()

    for n in [10_000, 100_000]:
        print(f"--- Dataset: {n:,} vectors ---")
        db = _random_unit_vectors(n, d, seed=0)
        queries = _random_unit_vectors(num_queries, d, seed=99)

        # Ground truth (brute-force float32)
        gt = np.argsort(-(queries @ db.T), axis=1)[:, :k]

        # Brute-force TurboQuant
        add_t, query_t, bf_idx = benchmark_brute_force(db, queries, k, num_bits)
        bf_recall = compute_recall(bf_idx, gt)
        bf_qps = num_queries / query_t
        print(
            f"  TurboQuantIndex (brute):  add={add_t:.2f}s  "
            f"query={query_t * 1000:.1f}ms  "
            f"recall@{k}={bf_recall:.3f}  QPS={bf_qps:.0f}"
        )

        # IVF TurboQuant
        nlist = max(10, int(np.sqrt(n)))
        for nprobe in [1, 5, 10]:
            train_t, add_t, query_t, ivf_idx = benchmark_ivf(
                db, queries, k, num_bits, nlist, nprobe,
            )
            ivf_recall = compute_recall(ivf_idx, gt)
            ivf_qps = num_queries / query_t
            print(
                f"  IVFTurboQuant (nlist={nlist}, nprobe={nprobe:2d}): "
                f"train={train_t:.2f}s  add={add_t:.2f}s  "
                f"query={query_t * 1000:.1f}ms  "
                f"recall@{k}={ivf_recall:.3f}  QPS={ivf_qps:.0f}"
            )

        print()

    print("Note: Query time includes all queries. QPS = queries per second.")
    print("Note: Recall is measured against float32 brute-force ground truth.")


if __name__ == "__main__":
    main()
