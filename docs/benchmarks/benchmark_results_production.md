# Solver Benchmark Report

Generated: 2026-02-28T01:47:38

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\014202_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `<REPO_ROOT>\benchmark_results\runs\2026_02_28\014202_solver_benchmarks\run_manifest.json`
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
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.09ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(0.97ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.92ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Solve<br>(1.05ms) | PE-Quad-Coupled<br>(1.7e-03) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.97ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.24ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.08ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(0.95ms) | PE-Quad-Coupled<br>(4.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.01ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.96ms) | T-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.07ms) | T-Solve<br>(1.7e-03) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.03ms) | PE-Quad-Coupled<br>(1.2e-02) | **T-Solve** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.64ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.78ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.11ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.43ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.49ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.55ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | PE-Quad-Coupled<br>(1.76ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(3.45ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.69ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.97ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(3.07ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.16ms) | T-Solve<br>(1.7e-03) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.84ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.17ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.24ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.08ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(1.58ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.86ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.16ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.75ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.77ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.14ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.09ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.56ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.62ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.27ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.97ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.23ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.42ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(1.83ms) | T-EVD-Solve<br>(1.6e-03) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.26ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(1.87ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(1.68ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Primal-Gram<br>(2.39ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(1.63ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(1.77ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.22ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(1.89ms) | T-EVD-Solve<br>(1.7e-03) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.67ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(4.01ms) | T-EVD-Solve<br>(1.7e-03) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 2.782e+00 | 1.296 | 1.653e-03 | 1.661e-03 | 0.0% | 2.795e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 3.267e+00 | 1.062 | 1.658e-03 | 1.663e-03 | 0.0% | 3.277e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 3.187e+00 | 1.028 | 1.661e-03 | 1.668e-03 | 0.0% | 3.200e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 3.528e+00 | 1.047 | 1.664e-03 | 1.668e-03 | 0.0% | 3.536e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.708e+00 | 1.523 | 4.385e-03 | 4.520e-03 | 0.0% | 1.761e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.298e+00 | 1.599 | 4.788e-03 | 6.180e-03 | 0.0% | 1.675e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.406e+00 | 1.755 | 4.899e-03 | 5.057e-03 | 0.0% | 1.451e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | Torch-Solve | 7.374e-01 | 3.019 | 4.714e-03 | 5.204e-03 | 0.0% | 8.141e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 8.594e-01 | 2.803 | 5.497e-03 | 5.540e-03 | 0.0% | 8.661e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 7.543e-01 | 2.705 | 4.938e-03 | 5.958e-03 | 0.0% | 9.101e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 4.479e-01 | 6.393 | 1.665e-03 | 1.667e-03 | 0.0% | 4.484e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | Torch-Solve | 2.888e-01 | 6.389 | 1.186e-02 | 1.276e-02 | 0.0% | 3.107e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.455e+01 | 1.635 | 1.662e-03 | 1.666e-03 | 0.0% | 1.459e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.236e+01 | 1.778 | 1.660e-03 | 1.664e-03 | 0.0% | 1.239e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.629e+00 | 2.110 | 1.662e-03 | 1.663e-03 | 0.0% | 4.632e+00 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.657e+00 | 2.429 | 1.662e-03 | 1.664e-03 | 0.0% | 4.663e+00 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.433e+01 | 1.491 | 1.654e-03 | 1.701e-03 | 0.0% | 1.474e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.435e+01 | 1.550 | 1.672e-03 | 1.713e-03 | 0.0% | 1.470e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.736e+00 | 1.831 | 1.666e-03 | 1.673e-03 | 0.0% | 3.752e+00 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 5.127e+00 | 3.449 | 1.667e-03 | 1.679e-03 | 0.0% | 5.164e+00 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.809e+00 | 1.685 | 1.658e-03 | 1.664e-03 | 0.0% | 4.826e+00 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.800e+00 | 1.970 | 1.662e-03 | 1.666e-03 | 0.0% | 4.812e+00 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.307e+00 | 3.369 | 1.660e-03 | 1.662e-03 | 0.0% | 1.309e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.308e+00 | 3.463 | 1.660e-03 | 1.661e-03 | 0.0% | 1.309e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.478e+00 | 1.753 | 2.828e-03 | 2.934e-03 | 0.0% | 4.646e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.338e+00 | 1.844 | 2.682e-03 | 2.717e-03 | 0.0% | 6.421e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 3.074e+00 | 2.169 | 2.936e-03 | 2.948e-03 | 0.0% | 3.087e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.653e+00 | 2.239 | 2.878e-03 | 3.158e-03 | 0.0% | 2.911e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.499e+00 | 1.748 | 2.657e-03 | 2.822e-03 | 0.0% | 5.840e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.087 | 2.136e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.499e+00 | 1.583 | 2.712e-03 | 2.858e-03 | 0.0% | 5.795e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.624e+00 | 1.856 | 2.660e-03 | 2.996e-03 | 0.0% | 5.208e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.899 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.149e+00 | 1.746 | 2.674e-03 | 2.705e-03 | 0.0% | 5.209e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.323e+00 | 1.769 | 2.714e-03 | 2.973e-03 | 0.0% | 4.736e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.944 | 2.185e-02 | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.278e+00 | 2.091 | 2.654e-03 | 2.949e-03 | 0.0% | 4.753e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.289e-01 | 3.556 | 4.551e-03 | 6.256e-03 | 0.0% | 1.002e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.054e+00 | 3.622 | 7.075e-03 | 7.079e-03 | 0.0% | 1.055e+00 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.937e+00 | 2.270 | 1.901e-03 | 1.907e-03 | 0.0% | 4.953e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.731e+00 | 1.970 | 1.901e-03 | 1.917e-03 | 0.0% | 6.788e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.244e+00 | 2.227 | 2.929e-03 | 3.630e-03 | 0.0% | 2.781e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.406e+00 | 2.505 | 1.896e-03 | 1.900e-03 | 0.0% | 2.411e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.875e+00 | 1.830 | 1.856e-03 | 1.949e-03 | 0.0% | 6.169e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.720 | 1.123e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.959e+00 | 1.869 | 1.907e-03 | 1.963e-03 | 0.0% | 6.134e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.486e+00 | 1.682 | 1.900e-03 | 1.905e-03 | 0.0% | 5.500e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.392 | 1.172e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.476e+00 | 1.627 | 1.908e-03 | 1.916e-03 | 0.0% | 5.499e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.993e+00 | 1.770 | 1.904e-03 | 1.918e-03 | 0.0% | 5.030e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.394 | 1.154e-02 | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.990e+00 | 1.892 | 1.904e-03 | 1.918e-03 | 0.0% | 5.027e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 1.060e+00 | 3.669 | 3.649e-03 | 3.736e-03 | 0.0% | 1.085e+00 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.017e+00 | 4.012 | 3.796e-03 | 3.803e-03 | 0.0% | 1.019e+00 |
