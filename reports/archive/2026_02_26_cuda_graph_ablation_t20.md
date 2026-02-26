# CUDA Graph Ablation for Coupled Solve (20 Trials)

*Date: 2026-02-26*

## Goal

Reduce non-GEMM overhead in `PE-Quad-Coupled-Apply` by replaying fixed-shape iterations via CUDA graphs.

## Implementation

- Added optional CLI flags in `scripts/matrix_solve.py`:
  - `--cuda-graph`
  - `--cuda-graph-warmup`
- Added timing-stability control:
  - `--timing-warmup-reps` (untimed warmup calls before measurement)
- Added a graph-capture timing path in `scripts/bench_solve_core.py`:
  - Applies to `PE-Quad-Coupled-Apply` on CUDA when `online_stop_tol is None`.
  - Falls back to eager execution on capture failure.

## Benchmark Suite

- Harness: `scripts/matrix_solve.py`
- Matrix: `n=1024`
- Cases: `{gaussian_spd, illcond_1e6}`
- Exponents: `p in {1,2,4}`
- RHS widths: `k in {1,16,64}`
- Precision/precond: `bf16`, `precond=jacobi`, `l_target=0.05`
- Trials: `20`, timing reps: `5`, timing warmup reps: `2`
- Online mode: `greedy-affine-opt`

Artifacts:
- Raw logs and parsed summary: `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/`
- Aggregate table: `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/summary_coupled_apply.md`
- Balanced paired validation (same inputs, primed): `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/paired_balanced_primed.md`

## Results

From the corrected 18-cell coupled-apply comparison (`on` vs `off`):

- `iter_ms`: **-3.59%** (faster)
- `relerr`: **+0.00%** (unchanged)
- `total_ms`: **-2.76%** (faster)
- Cell wins by total time: `on=12`, `off=6`

Interpretation:
- CUDA graph replay improves both steady-state iteration and overall coupled-apply totals under corrected timing methodology.
- Earlier mixed-signal totals were traced to first-run/order bias; adding untimed warmups and balanced paired checks removes that artifact.

## Bottleneck Evidence

Targeted profiler run (`p=2, k=16`) confirms launch-overhead reduction:

- Median per-call time: `1.973760 ms -> 1.900589 ms` (`-3.71%`)
- Profiler self CUDA total (20 calls): `44.718 ms -> 37.281 ms` (`-16.63%`)
- Launch API:
  - Off: `cudaLaunchKernel` `500` calls
  - On: `cudaGraphLaunch` `20` calls

Profiler artifacts:
- `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/profile/off_vs_on_profile.txt`
- `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/profile/summary.md`

## Recommendation

- Keep CUDA graph support **enabled for fixed-shape steady-state benchmarking**.
- Keep it **optional** at runtime for now (capture/warmup still adds complexity), but it is now a net-positive optimization under robust measurement.
