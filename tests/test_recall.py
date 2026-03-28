"""Test recall comparison: MSE-only vs MSE+QJL."""
import numpy as np
import time


def test_recall_comparison():
    """Compare MSE-only vs MSE+QJL recall on random unit vectors."""
    from turboquant import TurboQuantIndex

    d = 384
    n_db = 5000
    n_query = 100
    top_k = 10

    rng = np.random.RandomState(42)
    db = rng.randn(n_db, d).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    queries = rng.randn(n_query, d).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Ground truth
    gt_sims = queries @ db.T
    gt_top = np.argsort(-gt_sims, axis=1)[:, :top_k]

    print(f"\n{'='*60}")
    print(f"Recall@{top_k} Comparison (d={d}, db={n_db}, queries={n_query})")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Recall@10':<12} {'Cos Sim':<12} {'Compression':<12} {'Index ms':<10}")
    print(f"{'-'*76}")

    for bits in [3, 4, 5]:
        for use_qjl in [False, True]:
            label = f"{bits}-bit {'QJL' if use_qjl else 'MSE'}"
            if use_qjl and bits < 2:
                continue

            idx = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=use_qjl)

            t0 = time.time()
            idx.add(db)
            index_ms = (time.time() - t0) * 1000

            sims, indices = idx.search(queries, k=top_k)

            recalls = []
            for i in range(n_query):
                gt_set = set(gt_top[i])
                tq_set = set(indices[i])
                recalls.append(len(gt_set & tq_set) / top_k)

            avg_recall = np.mean(recalls)
            avg_cos = np.mean(np.sum(db * idx._reconstructed, axis=1))

            print(f"{label:<30} {avg_recall*100:<12.1f} {avg_cos:<12.4f} {idx.compression_ratio:<12.1f}x {index_ms:<10.0f}")

    print()


if __name__ == "__main__":
    test_recall_comparison()
