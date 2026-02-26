# Preconditioner + Gram-Path Ablation (`ideas/4`)

*Date: 2026-02-26*

## Scope

This pass evaluates new preconditioner modes from `ideas/4`:

- `jacobi` (diagonal normalization),
- `ruiz` (iterative symmetric equilibration),
- Gram-entry API `precond_gram_spd(...)` for `A = G^T G`.

Primary target remains practical GPU wall-clock.

## 1) Solve Harness (`matrix_solve.py`) — 20 Trials

- Setup: `p in {1,2,4}`, `k in {1,16,64}`, `n=1024`, cases `{gaussian_spd, illcond_1e6}`, dtype `bf16`.
- Metric used for default selection: `PE-Quad-Coupled-Apply` total latency.
- Raw logs: `benchmark_results/2026_02_26/idea4_precond_t20/`
- Parsed summary: `benchmark_results/2026_02_26/idea4_precond_t20/summary_coupled_apply.md`

Key result vs `frob` baseline:

- `jacobi`: **-1.55% total**, **-0.57% iter**, **-12.64% relerr** (overall means).
- Per-`p` total means: `jacobi` beats `frob` for `p=1,2,4`.

Decision:

- `benchmarks/solve/matrix_solve.py` default changed to `--precond jacobi`.

## 2) IRoot Harness (`matrix_iroot.py`) — 20 Trials

- Setup: `p in {1,2,4}`, `n=1024`, cases `{gaussian_spd, illcond_1e6, illcond_1e12, near_rank_def, spike}`, dtype `bf16`.
- Metric used for comparison: `PE-Quad-Coupled` total latency.
- Raw logs: `benchmark_results/2026_02_26/idea4_precond_iroot_t20/`
- Parsed summary: `benchmark_results/2026_02_26/idea4_precond_iroot_t20/summary_pe_quad_coupled.md`

Observed behavior:

- Overall mean total favors `jacobi` (`-2.29%` vs `frob`), but per-`p` is mixed.
- For `p=4` (current default exponent in `matrix_iroot.py`), `frob` is fastest in this sweep.

Decision:

- `benchmarks/inverse_root/matrix_iroot.py` default set to `--precond frob`.
- For explicit `p=2` benchmarking, `--precond jacobi` is currently the faster option.

## 3) Gram Path Check (`precond_gram_spd`)

- Setup: `G in R^(4096 x 1024)`, `bf16`, CUDA, `20` trials with alternating run order + warmup.
- Artifact: `benchmark_results/2026_02_26/idea4_gram_precond_t20/summary.md`

Result:

- `precond_gram_spd(gram_mode=\"col-norm\", mode=\"none\")` is numerically identical to
  `(G^T G)` then `precond_spd(mode=\"jacobi\")` in this check (`0` relative Frobenius diff).
- Runtime is close (within single-digit percent in this environment), with no extra
  algorithmic risk for the new API path.

## Final Defaults After This Pass

- Solve harness default preconditioner: **`jacobi`**
- IRoot harness default preconditioner: **`frob`** (aligned with default `p=4`)

