# Scripts and Tools

This directory contains the core benchmarking and verification tools for the Fast Matrix Inverse Roots project.

## 1. Benchmarking Suite (`scripts/bench/`)
Tools for generating benchmark matrices and summarizing results into a coherent report.
- `generate_benchmark_matrices.py`: Generates standardized SPD matrices for performance evaluation.
- `summarize.py`: Processes CSV results from `benchmark_poly_precond.py` to generate the final `BENCHMARK_REPORT.md`.

## 2. Verification (`scripts/`)
- `verify_phase2_policy.py`: Proves the mathematical and hardware-level correctness of the 2-step Phase 2 local refinement protocol. It demonstrates the convergence from Phase 1 output directly into the `bf16` machine epsilon noise floor using native CUDA GEMM operations.

---

## Polynomial Design Tools (`coeffs/`)
While the core Phase 2 polynomials are hardcoded into `fast_iroot/eval.py` for maximum performance, the tools used to design them are available in the `coeffs/` directory:
- `design_bf16_poly.py`: Designing and verifying Phase 1 global minimax polynomials.
- `design_local_poly.py`: Designing Phase 2 local minimax polynomials with exact `bf16` representable grid constraints.
