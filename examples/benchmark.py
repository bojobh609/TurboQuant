"""TurboQuant vs FAISS Benchmark.

Compares TurboQuant with FAISS across different index types on recall,
memory usage, and build/search speed.

Requirements: pip install turboquant[bench]
"""

import numpy as np
import time
import sys

from turboquant import TurboQuantIndex


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def benchmark_turboquant(db, queries, gt_top, k, bits):
    """Benchmark TurboQuant at given bit-width."""
    t0 = time.time()
    idx = TurboQuantIndex(dimension=db.shape[1], num_bits=bits, use_qjl=False)
    idx.add(db)
    build_ms = (time.time() - t0) * 1000

    t0 = time.time()
    sims, indices = idx.search(queries, k=k)
    search_ms = (time.time() - t0) * 1000

    recalls = [len(set(gt_top[i]) & set(indices[i])) / k for i in range(len(queries))]
    memory_mb = idx.size * idx._quantizer.bytes_per_vector / (1024 * 1024)

    return {
        "recall": np.mean(recalls),
        "build_ms": build_ms,
        "search_ms": search_ms,
        "memory_mb": memory_mb,
        "compression": idx.compression_ratio,
    }


def benchmark_faiss(db, queries, gt_top, k, index_type="flat"):
    """Benchmark FAISS index."""
    try:
        import faiss
    except ImportError:
        return None

    d = db.shape[1]
    t0 = time.time()

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, min(64, db.shape[0] // 10))
        index.train(db)
        index.nprobe = 10
    elif index_type == "pq":
        index = faiss.IndexPQ(d, 48, 8)  # 48 subquantizers, 8 bits each
        index.train(db)
    else:
        return None

    index.add(db)
    build_ms = (time.time() - t0) * 1000

    t0 = time.time()
    sims, indices = index.search(queries, k)
    search_ms = (time.time() - t0) * 1000

    recalls = [len(set(gt_top[i]) & set(indices[i])) / k for i in range(len(queries))]
    memory_mb = db.shape[0] * d * 4 / (1024 * 1024)  # approximate

    return {
        "recall": np.mean(recalls),
        "build_ms": build_ms,
        "search_ms": search_ms,
        "memory_mb": memory_mb,
    }


def main():
    d = 384
    n_db = 50_000
    n_query = 200
    k = 10

    print(f"TurboQuant vs FAISS Benchmark")
    print(f"{'='*70}")
    print(f"Database: {n_db:,} vectors, d={d}")
    print(f"Queries: {n_query}")
    print(f"Metric: Recall@{k}")
    print()

    print("Generating data...")
    db = _random_unit_vectors(n_db, d, seed=42)
    queries = _random_unit_vectors(n_query, d, seed=99)

    # Ground truth
    gt_sims = queries @ db.T
    gt_top = np.argsort(-gt_sims, axis=1)[:, :k]

    # Results table
    results = []

    # TurboQuant benchmarks
    for bits in [4, 5, 6]:
        r = benchmark_turboquant(db, queries, gt_top, k, bits)
        results.append((f"TurboQuant {bits}-bit", r))

    # FAISS benchmarks
    for idx_type, label in [("flat", "FAISS Flat"), ("pq", "FAISS PQ")]:
        r = benchmark_faiss(db, queries, gt_top, k, idx_type)
        if r:
            results.append((label, r))

    # Print results
    print(f"\n{'Method':<22} {'Recall@10':<12} {'Build (ms)':<12} {'Search (ms)':<13} {'Memory (MB)':<12} {'Compress':<10}")
    print(f"{'-'*81}")

    for name, r in results:
        comp = f"{r.get('compression', 1.0):.1f}x" if "compression" in r else "1.0x"
        print(f"{name:<22} {r['recall']*100:<12.1f} {r['build_ms']:<12.0f} {r['search_ms']:<13.1f} {r['memory_mb']:<12.1f} {comp:<10}")

    print(f"\n* FAISS Flat is exact search (100% recall) — included as reference")


if __name__ == "__main__":
    main()
