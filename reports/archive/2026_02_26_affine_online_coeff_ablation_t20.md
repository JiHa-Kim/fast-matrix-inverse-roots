# Solve Online Affine Schedule Ablation (20 Trials)

*Date: 2026-02-26*

## Setup

- Workload: `scripts/matrix_solve.py` coupled apply (`PE-Quad-Coupled-Apply`)
- Matrix: `n=1024`, `k in {1,16,64}`, cases `{gaussian_spd, illcond_1e6}`
- Exponents: `p in {1,2,4}`
- Precision/precond: `bf16`, `precond=jacobi`, `l_target=0.05`
- Trials: `20`, timing reps per trial: `5`

Raw logs are in `benchmark_results/2026_02_26/idea_affine_online_t20/`.
Parsed summary is `benchmark_results/2026_02_26/idea_affine_online_t20/summary_coupled_apply.md`.

## Winner

- **Default winner: `greedy-affine-opt`**
- Vs current default `greedy-newton`:
  - mean total delta: **-3.69%**
  - mean iter delta: **-1.28%**
  - mean relative-error delta: **-0.24%**

## Notes

- `greedy-affine-opt` wins strongly on `p=1` and `p=2`, and remains better than
  `greedy-newton` on `p=4` in this suite.
- `greedy-minimax` is faster than `greedy-affine-opt` for `p=4`, but slower overall.
- The CLI default for `--online-coeff-mode` was updated to `greedy-affine-opt`.
