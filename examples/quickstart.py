"""TurboQuant Quick Start — Compress and search vectors in 5 lines."""

import numpy as np
from turboquant import TurboQuantIndex

# 1. Create index
index = TurboQuantIndex(dimension=384, num_bits=6)

# 2. Generate sample data (10K random unit vectors)
rng = np.random.RandomState(42)
database = rng.randn(10_000, 384).astype(np.float32)

# 3. Add to index (auto-normalizes)
index.add(database)
print(f"Index: {index.stats()}")

# 4. Search
queries = rng.randn(5, 384).astype(np.float32)
similarities, indices = index.search(queries, k=10)

print(f"\nTop-10 results for 5 queries:")
for i in range(5):
    print(f"  Query {i}: indices={indices[i][:5]}..., sims={similarities[i][:3].round(3)}...")

# 5. Save and reload
index.save("/tmp/turboquant_demo")
loaded = TurboQuantIndex.load("/tmp/turboquant_demo")
print(f"\nLoaded index: {loaded.size} vectors")
