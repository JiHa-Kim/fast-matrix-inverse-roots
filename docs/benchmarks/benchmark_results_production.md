# Solver Benchmark Report

Generated: 2026-02-28T02:15:34

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\020952_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\020952_solver_benchmarks\run_manifest.json`
- markdown: `True`
- only: ``
- out: `docs/benchmarks/benchmark_results_production.md`
- prod: `False`
- run_name: `solver_benchmarks`
- timing_reps: `10`
- timing_warmup_reps: `2`
- trials: `10`

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

## Non-Normal (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.05ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | T-Solve<br>(1.37ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.10ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.26ms) | PE-Quad-Coupled<br>(1.7e-03) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.28ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.11ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.16ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.15ms) | PE-Quad-Coupled<br>(4.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.00ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(2.04ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.15ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.14ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.44ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.96ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.07ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.20ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.00ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.26ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | PE-Quad-Coupled<br>(1.77ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(3.32ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.65ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.86ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(3.61ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.77ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.24ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.36ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.71ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.71ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.02ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.13ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(3.32ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.64ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.17ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.81ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.06ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.15ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(1.95ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.55ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.83ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.09ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.43ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.55ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.10ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.93ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.23ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(3.25ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.74ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.21ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.57ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.69ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.35ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(3.85ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.13ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(4.09ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 2.643e+00 | 1.234 | 1.653e-03 | 1.661e-03 | 0.0% | 2.656e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 2.659e+00 | 1.374 | 1.658e-03 | 1.663e-03 | 0.0% | 2.667e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 2.819e+00 | 1.206 | 1.661e-03 | 1.668e-03 | 0.0% | 2.831e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 2.809e+00 | 1.324 | 1.664e-03 | 1.668e-03 | 0.0% | 2.816e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.366e+00 | 2.076 | 4.385e-03 | 4.520e-03 | 0.0% | 1.408e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.087e+00 | 1.834 | 4.788e-03 | 6.180e-03 | 0.0% | 1.403e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.354e+00 | 1.828 | 4.899e-03 | 5.057e-03 | 0.0% | 1.398e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | Torch-Solve | 7.399e-01 | 3.061 | 4.714e-03 | 5.204e-03 | 0.0% | 8.168e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 7.535e-01 | 3.185 | 5.497e-03 | 5.540e-03 | 0.0% | 7.594e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 7.336e-01 | 2.788 | 4.938e-03 | 5.958e-03 | 0.0% | 8.851e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 4.437e-01 | 6.453 | 1.665e-03 | 1.667e-03 | 0.0% | 4.442e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | Torch-Solve | 2.854e-01 | 6.441 | 1.186e-02 | 1.276e-02 | 0.0% | 3.071e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.507e+01 | 1.438 | 1.662e-03 | 1.666e-03 | 0.0% | 1.511e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.173e+01 | 1.957 | 1.660e-03 | 1.664e-03 | 0.0% | 1.176e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.522e+00 | 2.066 | 1.662e-03 | 1.663e-03 | 0.0% | 4.525e+00 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.604e+00 | 2.200 | 1.662e-03 | 1.664e-03 | 0.0% | 4.610e+00 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.430e+01 | 1.998 | 1.654e-03 | 1.701e-03 | 0.0% | 1.471e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.426e+01 | 1.264 | 1.672e-03 | 1.713e-03 | 0.0% | 1.461e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.676e+00 | 1.847 | 1.666e-03 | 1.673e-03 | 0.0% | 3.691e+00 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.283e+00 | 3.322 | 1.667e-03 | 1.679e-03 | 0.0% | 4.314e+00 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.802e+00 | 1.654 | 1.658e-03 | 1.664e-03 | 0.0% | 4.819e+00 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.792e+00 | 1.856 | 1.662e-03 | 1.666e-03 | 0.0% | 4.804e+00 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.292e+00 | 3.739 | 1.660e-03 | 1.662e-03 | 0.0% | 1.294e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.267e+00 | 4.129 | 1.660e-03 | 1.661e-03 | 0.0% | 1.268e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.471e+00 | 2.244 | 2.828e-03 | 2.934e-03 | 0.0% | 4.639e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.314e+00 | 2.363 | 2.682e-03 | 2.717e-03 | 0.0% | 6.396e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.835e+00 | 2.715 | 2.936e-03 | 2.948e-03 | 0.0% | 2.847e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.109e+00 | 2.711 | 2.878e-03 | 3.158e-03 | 0.0% | 2.314e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.500e+00 | 2.023 | 2.657e-03 | 2.822e-03 | 0.0% | 5.842e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.103 | 2.136e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.519e+00 | 3.321 | 2.712e-03 | 2.858e-03 | 0.0% | 5.816e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.609e+00 | 1.635 | 2.660e-03 | 2.996e-03 | 0.0% | 5.191e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.910 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.113e+00 | 1.807 | 2.674e-03 | 2.705e-03 | 0.0% | 5.172e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.318e+00 | 2.060 | 2.714e-03 | 2.973e-03 | 0.0% | 4.730e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.958 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.275e+00 | 1.951 | 2.654e-03 | 2.949e-03 | 0.0% | 4.750e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.289e-01 | 3.552 | 4.551e-03 | 6.256e-03 | 0.0% | 1.002e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.015e+00 | 3.829 | 7.075e-03 | 7.079e-03 | 0.0% | 1.016e+00 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.939e+00 | 2.094 | 1.901e-03 | 1.907e-03 | 0.0% | 4.955e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.701e+00 | 2.427 | 1.901e-03 | 1.917e-03 | 0.0% | 6.757e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | Chebyshev-Apply | 2.281e+00 | 2.766 | 1.896e-03 | 1.904e-03 | 0.0% | 2.291e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.341e+00 | 3.116 | 1.896e-03 | 1.900e-03 | 0.0% | 2.346e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.879e+00 | 1.934 | 1.856e-03 | 1.949e-03 | 0.0% | 6.174e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.510 | 1.123e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.947e+00 | 3.255 | 1.907e-03 | 1.963e-03 | 0.0% | 6.122e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.481e+00 | 1.738 | 1.900e-03 | 1.905e-03 | 0.0% | 5.495e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.443 | 1.172e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.472e+00 | 1.568 | 1.908e-03 | 1.916e-03 | 0.0% | 5.495e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.908e+00 | 2.688 | 1.904e-03 | 1.918e-03 | 0.0% | 4.944e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.448 | 1.154e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.981e+00 | 3.851 | 1.904e-03 | 1.918e-03 | 0.0% | 5.018e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 9.836e-01 | 4.129 | 3.649e-03 | 3.736e-03 | 0.0% | 1.007e+00 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.013e+00 | 4.093 | 3.796e-03 | 3.803e-03 | 0.0% | 1.015e+00 |
