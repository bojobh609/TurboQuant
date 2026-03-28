"""Semantic Search with TurboQuant + Sentence Transformers.

Demonstrates building a compressed semantic search engine with 5x less memory
than FAISS FlatL2, while maintaining 95%+ recall.

Requirements: pip install turboquant[bench]
"""

import numpy as np
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install sentence-transformers: pip install turboquant[bench]")
    raise

from turboquant import TurboQuantIndex

# Sample corpus
documents = [
    "TurboQuant achieves near-optimal vector quantization for AI applications",
    "FAISS is a library for efficient similarity search and clustering of dense vectors",
    "Product quantization divides vectors into sub-vectors and quantizes each independently",
    "The Johnson-Lindenstrauss lemma states that points can be projected to lower dimensions",
    "Transformer models use self-attention mechanisms for sequence processing",
    "Vector databases store and retrieve high-dimensional embeddings efficiently",
    "Approximate nearest neighbor search trades accuracy for speed in large datasets",
    "Embedding models convert text into dense numerical representations",
    "Cosine similarity measures the angle between two vectors in high-dimensional space",
    "Random rotation matrices preserve distances between points with high probability",
    "Lloyd-Max algorithm finds optimal quantization levels for a given distribution",
    "The Shannon distortion-rate function gives theoretical limits on compression quality",
    "Semantic search understands query intent rather than matching exact keywords",
    "KV cache compression reduces memory requirements during LLM inference",
    "Information retrieval systems rank documents by relevance to user queries",
    "Dense retrieval uses neural embeddings instead of sparse term frequency vectors",
    "Retrieval-augmented generation combines search with language model generation",
    "Vector quantization reduces storage by mapping vectors to a finite codebook",
    "Sentence embeddings capture semantic meaning in fixed-dimensional vectors",
    "Nearest neighbor search finds the most similar items in a vector database",
]

print("Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode documents
print(f"Encoding {len(documents)} documents...")
doc_embeddings = model.encode(documents, normalize_embeddings=True)
print(f"  Embedding shape: {doc_embeddings.shape}")
print(f"  Embedding dtype: {doc_embeddings.dtype}")

# Build TurboQuant index
print("\nBuilding TurboQuant index (6-bit, 5.3x compression)...")
t0 = time.time()
index = TurboQuantIndex(dimension=384, num_bits=6)
index.add(doc_embeddings)
build_time = time.time() - t0
print(f"  Build time: {build_time*1000:.1f}ms")
print(f"  Stats: {index.stats()}")

# Search
queries = [
    "How does vector compression work?",
    "What is semantic similarity?",
    "How to build a search engine?",
]

print("\nSearch Results:")
print("=" * 70)

for query in queries:
    query_emb = model.encode([query], normalize_embeddings=True)

    t0 = time.time()
    similarities, doc_indices = index.search(query_emb, k=3)
    search_ms = (time.time() - t0) * 1000

    print(f"\nQuery: \"{query}\" ({search_ms:.1f}ms)")
    for rank, (sim, idx) in enumerate(zip(similarities[0], doc_indices[0])):
        print(f"  {rank+1}. [{sim:.3f}] {documents[idx]}")
