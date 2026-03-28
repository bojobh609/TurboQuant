# Contributing to TurboQuant

We welcome contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Firmamento-Technologies/TurboQuant.git
cd TurboQuant
pip install -e ".[dev]"
pytest tests/ -v
```

## What We're Looking For

- **Performance improvements** — Faster quantization/search, SIMD, batch operations
- **New bit-packing** — Pack 4-bit codes into uint8 (2 values per byte)
- **GPU acceleration** — Optional CuPy/PyTorch backend
- **FAISS adapter** — Use TurboQuant as a FAISS custom index
- **Benchmarks** — More datasets, higher dimensions, comparison with ScaNN/Annoy
- **Documentation** — Tutorials, API docs, mathematical explanations

## Guidelines

1. **Tests required** — All PRs must include tests. Run `pytest tests/ -v` before submitting.
2. **Pure Python** — Core algorithms must work with NumPy only. GPU/C++ extensions go in optional modules.
3. **Type hints** — All public functions must have type annotations.
4. **Docstrings** — Google-style docstrings for all public classes and methods.

## Code Style

- Python 3.10+ features (type unions with `|`, etc.)
- NumPy-style array operations (avoid Python loops on vectors)
- Keep modules focused — one responsibility per file

## Submitting

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests first, then implementation
4. Run `pytest tests/ -v` and ensure all pass
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
