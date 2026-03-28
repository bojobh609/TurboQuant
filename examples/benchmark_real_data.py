"""Benchmark TurboQuant recall on realistic data distributions.

Tests isotropic (best case), clustered (typical embeddings), and
anisotropic (fine-tuned models) distributions.

Run: python examples/benchmark_real_data.py
"""

import numpy as np
from turboquant import TurboQuantIndex


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def make_isotropic(n, d, seed=42):
    """Random unit vectors — best case for TurboQuant."""
    rng = np.random.RandomState(seed)
    return _normalize(rng.randn(n, d).astype(np.float32))


def make_clustered(n, d, n_clusters=20, seed=42):
    """Gaussian clusters — mimics sentence/document embeddings."""
    rng = np.random.RandomState(seed)
    centers = _normalize(rng.randn(n_clusters, d).astype(np.float32))
    vectors = []
    per_cluster = n // n_clusters
    for i in range(n_clusters):
        spread = 0.1 + rng.rand() * 0.3
        cluster = centers[i] + rng.randn(per_cluster, d).astype(np.float32) * spread
        vectors.append(cluster)
    return _normalize(np.concatenate(vectors, axis=0)[:n])


def make_anisotropic(n, d, rank=50, seed=42):
    """Low-rank structure — mimics fine-tuned model embeddings."""
    rng = np.random.RandomState(seed)
    basis = rng.randn(rank, d).astype(np.float32)
    coords = rng.randn(n, rank).astype(np.float32)
    vectors = coords @ basis
    vectors += rng.randn(n, d).astype(np.float32) * 0.05
    return _normalize(vectors)


def compute_recall_at_k(pred, gt, k):
    recalls = []
    for i in range(len(pred)):
        pred_set = set(int(x) for x in pred[i][:k])
        gt_set = set(int(x) for x in gt[i][:k])
        recalls.append(len(pred_set & gt_set) / k)
    return np.mean(recalls)


def benchmark_distribution(name, db, queries, bits_list):
    """Run recall benchmark for a given distribution."""
    gt = np.argsort(-(queries @ db.T), axis=1)

    print(f"\n### {name} (N={db.shape[0]}, d={db.shape[1]})")
    print(f"| Bits | Recall@1 | Recall@10 | Recall@100 |")
    print(f"|------|----------|-----------|------------|")

    for bits in bits_list:
        idx = TurboQuantIndex(dimension=db.shape[1], num_bits=bits, use_qjl=False)
        idx.add(db)
        _, pred = idx.search(queries, k=100)

        r1 = compute_recall_at_k(pred, gt, 1)
        r10 = compute_recall_at_k(pred, gt, 10)
        r100 = compute_recall_at_k(pred, gt, 100)
        print(f"| {bits}    | {r1:.1%}   | {r10:.1%}    | {r100:.1%}     |")


def main():
    d = 384
    n_db = 10_000
    n_queries = 200
    bits_list = [2, 3, 4, 5, 6, 8]

    print("=" * 70)
    print("TurboQuant Recall Benchmark — Multiple Data Distributions")
    print("=" * 70)
    print()
    print("Evaluating recall against exact float32 brute-force ground truth.")
    print("All vectors L2-normalized before indexing.")

    queries = make_isotropic(n_queries, d, seed=999)

    db_iso = make_isotropic(n_db, d, seed=0)
    benchmark_distribution("Isotropic (random unit vectors)", db_iso, queries, bits_list)

    db_clust = make_clustered(n_db, d, n_clusters=20, seed=0)
    benchmark_distribution("Clustered (20 Gaussian clusters)", db_clust, queries, bits_list)

    db_aniso = make_anisotropic(n_db, d, rank=50, seed=0)
    benchmark_distribution("Anisotropic (rank-50 subspace)", db_aniso, queries, bits_list)

    print()
    print("Note: Isotropic is the best case for TurboQuant (matches theoretical assumptions).")
    print("Clustered and anisotropic distributions better represent real-world embeddings.")
    print("Real embedding recall may differ from these synthetic benchmarks.")


if __name__ == "__main__":
    main()
