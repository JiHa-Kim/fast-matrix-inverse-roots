# Fast Matrix Inverse p-th Roots (GPU-Focused)

Fast, practical inverse p-th root iteration for SPD matrices, tuned for ML preconditioning workloads.

This project prioritizes:
- fixed small iteration budgets
- GEMM-dominated kernels (matmul-only, no solves/QR)
- bf16-friendly stability
- empirical benchmarking over purely theoretical comparisons

## Repository Layout

- `fast_iroot/`
  - `precond.py` — Preconditioning logic (`precond_spd`)
  - `coupled.py` — Coupled quadratic PE iterations (`inverse_sqrt_pe_quadratic`, `inverse_proot_pe_quadratic_coupled`)
  - `uncoupled.py` — Uncoupled quadratic PE iterations (`inverse_proot_pe_quadratic_uncoupled`)
  - `coeffs.py` — Coefficient schedule loading/tuning hooks (`build_pe_schedules`)
  - `metrics.py` — Quality metrics (`compute_quality_stats`, `exact_inverse_proot`)
  - `utils.py` — Low-level helpers (`_matmul_into`, `_addmm_into`, `_bpow_times_y`)
  - `auto_policy.py` — Legacy auto-policy utilities (currently unused)
- `matrix_iroot.py`
  - Main benchmark harness CLI for inverse p-th roots
- `coeff_tuner.py`
  - Offline schedule tuning utility
- `verify_iroot.py`
  - Correctness test across p∈{1,2,3,4,8}
- `results/`
  - Benchmark results and comprehensive report
- `archive/`
  - Archived affine/NS methods (deprecated, reference only)

## Environment

`pyproject.toml` is configured for `uv` and CUDA-enabled PyTorch wheels.

### Install

```bash
uv sync
```

## Quick Start

Run a quick benchmark (inverse 4th root):

```bash
uv run python matrix_iroot.py --p 4 --sizes 256,512 --dtype bf16 --trials 8
```

Run for matrix inverse (p=1):

```bash
uv run python matrix_iroot.py --p 1 --sizes 256,512,1024 --dtype bf16 --trials 8 --coeff-mode tuned
```

Verify correctness across multiple p values:

```bash
uv run python verify_iroot.py
```

## Methods

The project uses **quadratic polynomial-express (PE-Quad)** iterations exclusively:

### PE-Quad (Uncoupled)
Tracks only `X ≈ A^{-1/p}`, recomputing `Y = X^p · A` each step.
- Lower memory (5 workspace tensors)
- Works for any p

### PE-Quad-Coupled
Tracks both `X ≈ A^{-1/p}` and `Y ≈ A · X^p`.
- Terminal-step optimization: skips Y-update on last iteration (saves 2-3 matmuls)
- 10-14% faster iteration time for p≥2 at larger sizes
- Works for any p via binary exponentiation (`_bpow_times_y`)

### Deprecated Methods (archived)
Affine methods (PE-Affine, Newton-Schulz NS3/NS4, PE-NS3) are archived in `archive/affine_iterations.py`. They consistently underperform quadratic methods in both speed and residual quality.

## Important CLI Flags

```text
--p               Root exponent (1=inverse, 2=inv sqrt, 4=inv 4th root, etc.)
--sizes           Matrix dimensions to benchmark
--dtype {fp32,bf16}
--trials          Number of test matrices per case
--compile         Enable torch.compile
--precond {none,frob,aol}
--coeff-mode {auto,precomputed,tuned}
--coeff-seed      Seed for coefficient tuning
--coeff-safety    Safety scaling factor
--target-resid    Target residual threshold
--metrics-mode {full,fast}
--power-iters     Spectral residual estimation iterations
--mv-samples      Random MV probe sample count
--hard-probe-iters  Hard-direction probe iterations
```

## Metrics Reported

Per method and case:
- Total median ms (precond + iteration)
- Iteration median ms
- Residual (median / p95 / max)
- Relative error vs eigendecomp
- Symmetry diagnostics (symX, symW)
- Bad count (NaN/Inf)

## Results

See `results/benchmark_report.md` for the latest comprehensive benchmark data.

## Tuning Coefficients

Use `coeff_tuner.py` for offline schedule generation:
- Precomputed schedules available for p=2, l_target=0.05
- Tuned schedules for arbitrary p and targets via `--coeff-mode tuned`
- Optional safety scaling

## References

- Amsel et al., 2025. *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm* (arXiv:2505.16932)
- Boissin et al., 2025. *Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning* (arXiv:2512.04632)
