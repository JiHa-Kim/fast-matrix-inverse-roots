# Solver Benchmark Report

Generated: 2026-02-28T05:23:21

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\051603_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\051603_solver_benchmarks\run_manifest.json`
- markdown: `True`
- only: ``
- out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\docs\benchmarks\benchmark_results_production.md`
- prod: `True`
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
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.06ms) | T-Linalg-Solve<br>(3.4e-07) | **T-Linalg-Solve** |
| **256** / **256**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.06ms) | T-Linalg-Solve<br>(8.4e-08) | **T-Linalg-Solve** |
| **256** / **256**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.99ms) | T-Linalg-Solve<br>(4.6e-07) | **T-Linalg-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Linalg-Solve<br>(0.97ms) | T-Linalg-Solve<br>(4.4e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.07ms) | T-Linalg-Solve<br>(7.5e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.22ms) | T-Linalg-Solve<br>(6.6e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.13ms) | T-Linalg-Solve<br>(9.1e-05) | **T-Linalg-Solve** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(1.26ms) | T-Linalg-Solve<br>(4.4e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(2.00ms) | T-Linalg-Solve<br>(9.8e-05) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(2.03ms) | T-Linalg-Solve<br>(9.1e-05) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(2.00ms) | T-Linalg-Solve<br>(1.2e-04) | **T-Linalg-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.03ms) | T-Linalg-Solve<br>(1.2e-02) | **PE-Quad-Coupled** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.42ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.17ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.78ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.79ms) | T-Cholesky-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.75ms) | T-Cholesky-Solve<br>(3.5e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.84ms) | T-Cholesky-Solve<br>(3.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.36ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.66ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.60ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.32ms) | T-Cholesky-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(4.33ms) | T-Cholesky-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.98ms) | T-Cholesky-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.70ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.89ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.56ms) | T-Linalg-Solve<br>(3.8e-04) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | Chebyshev<br>(3.46ms) | T-Linalg-Solve<br>(3.9e-04) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.93ms) | T-Linalg-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.90ms) | T-Linalg-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.98ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.47ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.59ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.73ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.60ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | Inverse-Newton-Coupled<br>(5.68ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(3.11ms) | T-Linalg-Solve<br>(3.7e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.76ms) | T-Linalg-Solve<br>(3.7e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.37ms) | T-Linalg-Solve<br>(4.0e-04) | **Chebyshev** |
| **512** / **512**<br>`illcond_1e6` | Chebyshev<br>(3.31ms) | T-Linalg-Solve<br>(4.2e-04) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(3.05ms) | T-Linalg-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.61ms) | T-Linalg-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.58ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.66ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(3.54ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(3.11ms) | T-Linalg-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.41ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(5.12ms) | T-Linalg-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | resid | nf_rate | qf_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Linalg-Solve | 5.989e+00 | 1.204 | 3.407e-07 | 3.426e-07 | 0.0% | 0.0% | 6.304e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Linalg-Solve | 7.739e+00 | 1.135 | 8.427e-08 | 8.606e-08 | 0.0% | 0.0% | 7.779e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Linalg-Solve | 7.295e+00 | 1.018 | 4.613e-07 | 4.653e-07 | 0.0% | 0.0% | 7.546e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Linalg-Solve | 3.044e+00 | 0.969 | 4.431e-05 | 3.721e-04 | 0.0% | 10.0% | 5.601e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | Torch-Linalg-Solve | 1.328e+00 | 3.299 | 7.499e-05 | 7.819e-05 | 0.0% | 0.0% | 1.330e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | Torch-Linalg-Solve | 1.457e+00 | 3.066 | 6.601e-05 | 6.773e-05 | 0.0% | 0.0% | 1.468e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | Torch-Linalg-Solve | 1.426e+00 | 3.086 | 9.087e-05 | 1.037e-04 | 0.0% | 0.0% | 1.430e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 3.912 | 4.714e-03 | 4.179e+00 | 0.0% | 100.0% | 6.226e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | Torch-Linalg-Solve | 6.476e-01 | 6.348 | 9.804e-05 | 1.018e-04 | 0.0% | 0.0% | 6.485e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | Torch-Linalg-Solve | 6.517e-01 | 6.323 | 9.095e-05 | 9.266e-05 | 0.0% | 0.0% | 6.635e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Linalg-Solve | 6.364e-01 | 6.332 | 1.212e-04 | 1.376e-04 | 0.0% | 0.0% | 6.369e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 7.349 | 1.186e-02 | 5.537e+00 | 0.0% | 100.0% | 2.710e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 4.140e+01 | 2.422 | 1.642e-07 | 1.201e-04 | 0.0% | 0.0% | 4.286e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 4.078e+01 | 2.167 | 1.647e-07 | 1.559e-04 | 0.0% | 0.0% | 4.246e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.126e+01 | 2.782 | 1.663e-07 | 7.764e-05 | 0.0% | 0.0% | 1.171e+01 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.148e+01 | 2.785 | 1.635e-07 | 1.156e-04 | 0.0% | 0.0% | 1.169e+01 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.325e+01 | 2.745 | 3.513e-07 | 4.312e-05 | 0.0% | 0.0% | 3.441e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.327e+01 | 1.840 | 3.573e-07 | 8.167e-05 | 0.0% | 0.0% | 3.408e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.204e+01 | 2.361 | 1.676e-07 | 4.333e-05 | 0.0% | 0.0% | 1.238e+01 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.184e+01 | 2.659 | 1.679e-07 | 8.223e-05 | 0.0% | 0.0% | 1.211e+01 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.151e+01 | 2.604 | 1.677e-07 | 4.339e-05 | 0.0% | 0.0% | 1.174e+01 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.146e+01 | 2.321 | 1.681e-07 | 8.176e-05 | 0.0% | 0.0% | 1.168e+01 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.217e+00 | 4.602 | 1.891e-07 | 4.285e-05 | 0.0% | 0.0% | 3.256e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.202e+00 | 4.341 | 1.870e-07 | 8.165e-05 | 0.0% | 0.0% | 3.250e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.469e+00 | 2.698 | 2.828e-03 | 2.834e-03 | 0.0% | 0.0% | 4.637e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.328e+00 | 2.889 | 2.682e-03 | 2.686e-03 | 0.0% | 0.0% | 6.411e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.619e+00 | 3.558 | 2.936e-03 | 2.942e-03 | 0.0% | 0.0% | 2.630e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.061e+00 | 3.458 | 2.878e-03 | 2.881e-03 | 0.0% | 0.0% | 2.261e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.495e+00 | 2.932 | 2.657e-03 | 2.659e-03 | 0.0% | 0.0% | 5.836e+00 |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.485e+00 | 2.895 | 2.712e-03 | 2.714e-03 | 0.0% | 0.0% | 5.780e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 4.626e+00 | 2.975 | 2.660e-03 | 2.662e-03 | 0.0% | 0.0% | 5.210e+00 |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.125e+00 | 2.467 | 2.674e-03 | 2.675e-03 | 0.0% | 0.0% | 5.184e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.326e+00 | 2.590 | 2.714e-03 | 2.715e-03 | 0.0% | 0.0% | 4.739e+00 |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.198e+00 | 2.729 | 2.654e-03 | 2.655e-03 | 0.0% | 0.0% | 4.665e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.682e-01 | 4.596 | 4.551e-03 | 4.557e-03 | 0.0% | 0.0% | 1.056e+00 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 6.111e-01 | 6.264 | 7.075e-03 | 7.079e-03 | 0.0% | 0.0% | 6.114e-01 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.938e+00 | 3.112 | 1.901e-03 | 1.902e-03 | 0.0% | 0.0% | 4.954e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.728e+00 | 2.763 | 1.901e-03 | 1.901e-03 | 0.0% | 0.0% | 6.785e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | Chebyshev-Apply | 2.311e+00 | 3.441 | 1.896e-03 | 1.897e-03 | 0.0% | 0.0% | 2.321e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.416e+00 | 3.311 | 1.896e-03 | 1.896e-03 | 0.0% | 0.0% | 2.421e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 5.886e+00 | 3.048 | 1.856e-03 | 1.856e-03 | 0.0% | 0.0% | 6.181e+00 |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 5.947e+00 | 2.606 | 1.907e-03 | 1.908e-03 | 0.0% | 0.0% | 6.122e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 5.471e+00 | 2.583 | 1.900e-03 | 1.900e-03 | 0.0% | 0.0% | 5.485e+00 |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 5.455e+00 | 2.656 | 1.908e-03 | 1.909e-03 | 0.0% | 0.0% | 5.478e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 4.978e+00 | 3.535 | 1.904e-03 | 1.904e-03 | 0.0% | 0.0% | 5.015e+00 |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.992e+00 | 3.111 | 1.904e-03 | 1.904e-03 | 0.0% | 0.0% | 5.029e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 9.855e-01 | 4.408 | 3.649e-03 | 3.651e-03 | 0.0% | 0.0% | 1.009e+00 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 9.381e-01 | 5.125 | 3.796e-03 | 3.801e-03 | 0.0% | 0.0% | 9.398e-01 |
