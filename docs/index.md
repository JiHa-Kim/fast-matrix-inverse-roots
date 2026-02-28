# Fast Matrix Inverse Roots Documentation

Welcome to the documentation for `fast-matrix-inverse-roots`. This project provides production-oriented inverse $p$-th-root and inverse-apply kernels for PyTorch workloads, optimized for machine learning systems.

## Key Sections

- **[API Reference](api.md)**: Detailed documentation of the public API, including high-level solvers and configuration objects.
- **[Methods & Architecture](methods/README.md)**: Deep dives into the mathematical methods (PE-Quad, Chebyshev, etc.) and the overall system architecture.
- **[Benchmark Decisions](benchmark_decisions.md)**: A log of architectural decisions driven by empirical benchmark results.
- **[Implementation Status](implementation_status.md)**: A summary of what is currently implemented, partially implemented, or planned.
- **[Glossary](glossary.md)**: A guide to abbreviations and technical terms used in the project.
- **[Roadmap](roadmap.md)**: Upcoming features and priority tasks.
- **[Guidelines & Contributing](guidelines.md)**: Repository standards, coding style, and testing requirements.

## Getting Started

If you are new to the project, start with the [Quickstart](../README.md#quickstart) in the main README.

### Installation

```bash
uv sync
```

### Core Philosophy

The primary goal of `fast_iroot` is to compute $Z \approx A^{-1/p} B$ efficiently without necessarily materializing the dense matrix $A^{-1/p}$. It leverages:
- **Coupled Quadratic PE Kernels**: For fast convergence on SPD matrices.
- **Gram-Matrix Optimization**: Specialized paths for $A = G^T G$ or $A = G G^T$ that avoid $O(n^3)$ costs where possible.
- **Workspace Reuse**: Minimal allocation overhead for repeated solves in ML loops.
