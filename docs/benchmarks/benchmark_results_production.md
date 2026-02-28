# Solver Benchmark Report

Generated: 2026-02-28T03:53:53

## Run Configuration

- ab_baseline_rows_in: ``
- ab_extra_args_a: ``
- ab_extra_args_b: ``
- ab_interleave: `True`
- ab_label_a: `A`
- ab_label_b: `B`
- ab_match_on_method: `True`
- ab_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\034634_solver_benchmarks\solver_benchmarks_ab.md`
- baseline_rows_out: ``
- dtype: `bf16`
- extra_args: ``
- integrity_checksums: `True`
- manifest_out: `D:\GitHub\JiHa-Kim\fast-matrix-inverse-roots\benchmark_results\runs\2026_02_28\034634_solver_benchmarks\run_manifest.json`
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
| **256** / **256**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(0.96ms) | T-Solve<br>(3.4e-07) | **T-Solve** |
| **256** / **256**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.05ms) | T-Solve<br>(8.4e-08) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(0.99ms) | T-Solve<br>(4.6e-07) | **T-Solve** |
| **256** / **256**<br>`similarity_posspec_hard` | T-Solve<br>(1.12ms) | T-Solve<br>(4.4e-05) | **T-Solve** |
| **512** / **512**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.07ms) | T-Solve<br>(7.5e-05) | **PE-Quad-Coupled** |
| **512** / **512**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.03ms) | T-Solve<br>(6.6e-05) | **T-Solve** |
| **512** / **512**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.12ms) | T-Solve<br>(9.1e-05) | **PE-Quad-Coupled** |
| **512** / **512**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(0.97ms) | T-Solve<br>(4.4e-03) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`gaussian_shifted` | Inverse-Newton-Coupled<br>(1.91ms) | T-Solve<br>(9.8e-05) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`nonnormal_upper` | Inverse-Newton-Coupled<br>(1.92ms) | T-Solve<br>(9.1e-05) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`similarity_posspec` | Inverse-Newton-Coupled<br>(1.97ms) | T-Solve<br>(1.2e-04) | **T-Solve** |
| **1024** / **1024**<br>`similarity_posspec_hard` | Inverse-Newton-Coupled<br>(2.07ms) | T-Solve<br>(1.2e-02) | **PE-Quad-Coupled** |

## SPD (p=1)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(1.75ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **256** / **256**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.95ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.29ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **512** / **512**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.88ms) | T-Solve<br>(1.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.11ms) | T-Solve<br>(3.5e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.89ms) | T-Solve<br>(3.6e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.08ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **16**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(1.84ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`gaussian_spd` | T-Cholesky-Solve-R<br>(2.02ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **64**<br>`illcond_1e6` | T-Cholesky-Solve-R<br>(2.00ms) | T-Solve<br>(1.7e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`gaussian_spd` | Inverse-Newton-Coupled<br>(3.29ms) | T-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(3.23ms) | T-Solve<br>(1.9e-07) | **T-Cholesky-Solve-R** |

## SPD (p=2)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(1.86ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(2.19ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.73ms) | T-EVD-Solve<br>(3.8e-04) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.72ms) | T-EVD-Solve<br>(3.9e-04) | **PE-Quad-Coupled** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.05ms) | T-EVD-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.10ms) | PE-Quad-Coupled-Primal-Gram<br>(2.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(3.35ms) | T-EVD-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.99ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.11ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.31ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.99ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.06ms) | PE-Quad-Coupled-Primal-Gram<br>(2.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(4.74ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(3.61ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(5.62ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## SPD (p=4)

| Problem Scenario | Fastest Method | Most Accurate | Overall Winner |
|:---|:---|:---|:---|
| **256** / **256**<br>`gaussian_spd` | Chebyshev<br>(2.02ms) | T-EVD-Solve<br>(3.7e-04) | **Chebyshev** |
| **256** / **256**<br>`illcond_1e6` | Chebyshev<br>(1.79ms) | T-EVD-Solve<br>(3.7e-04) | **Chebyshev** |
| **512** / **512**<br>`gaussian_spd` | PE-Quad-Coupled<br>(2.74ms) | T-EVD-Solve<br>(4.0e-04) | **PE-Quad-Coupled** |
| **512** / **512**<br>`illcond_1e6` | PE-Quad-Coupled<br>(2.41ms) | T-EVD-Solve<br>(4.2e-04) | **Chebyshev** |
| **1024** / **1**<br>`gaussian_spd` | Chebyshev<br>(2.31ms) | T-EVD-Solve<br>(1.7e-06) | **Chebyshev** |
| **1024** / **1**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.19ms) | PE-Quad-Coupled-Primal-Gram<br>(1.1e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **1**<br>`illcond_1e6` | Chebyshev<br>(2.25ms) | T-EVD-Solve<br>(1.4e-06) | **Chebyshev** |
| **1024** / **16**<br>`gaussian_spd` | Chebyshev<br>(2.93ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **16**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.28ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **16**<br>`illcond_1e6` | Chebyshev<br>(2.85ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gaussian_spd` | Chebyshev<br>(2.53ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **64**<br>`gram_rhs_gtb` | PE-Quad-Coupled-Dual-Gram<br>(1.19ms) | PE-Quad-Coupled-Primal-Gram<br>(1.2e-02) | **PE-Quad-Coupled-Primal-Gram** |
| **1024** / **64**<br>`illcond_1e6` | Chebyshev<br>(2.90ms) | T-EVD-Solve<br>(3.6e-04) | **Chebyshev** |
| **1024** / **1024**<br>`gaussian_spd` | PE-Quad-Coupled<br>(4.73ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |
| **1024** / **1024**<br>`illcond_1e6` | PE-Quad-Coupled<br>(6.40ms) | T-EVD-Solve<br>(3.6e-04) | **PE-Quad-Coupled** |

## Legend

- **Scenario**: Matrix size (n) / RHS dimension (k) / Problem case.
- **Fastest**: Method with lowest execution time.
- **Most Accurate**: Method with lowest median relative error.
- **Overall Winner**: Optimal balance of speed and quality (highest assessment score).

---

### Detailed Assessment Leaders

| kind | p | n | k | case | best_method | score | total_ms | relerr | resid | relerr_p90 | fail_rate | q_per_ms |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | Torch-Solve | 6.245e+00 | 1.143 | 3.407e-07 | 3.426e-07 | 3.586e-07 | 0.0% | 6.573e+00 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | Torch-Solve | 6.833e+00 | 1.312 | 8.427e-08 | 8.606e-08 | 8.471e-08 | 0.0% | 6.869e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec | Torch-Solve | 7.257e+00 | 1.009 | 4.613e-07 | 4.653e-07 | 4.772e-07 | 0.0% | 7.507e+00 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | Torch-Solve | 3.047e+00 | 1.123 | 4.431e-05 | 3.721e-04 | 7.338e-05 | 10.0% | 5.606e+00 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.775e+00 | 1.486 | 4.385e-03 | 4.744e-03 | 4.520e-03 | 0.0% | 1.830e+00 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | Torch-Solve | 1.494e+00 | 2.998 | 6.601e-05 | 6.773e-05 | 6.650e-05 | 0.0% | 1.505e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 1.607e+00 | 1.675 | 4.899e-03 | 6.315e-03 | 5.057e-03 | 0.0% | 1.659e+00 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 3.610 | 4.714e-03 | 4.179e+00 | 5.204e-03 | 100.0% | 6.731e-01 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 8.041e-01 | 2.956 | 5.497e-03 | 5.973e-03 | 5.540e-03 | 0.0% | 8.104e-01 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 7.947e-01 | 2.582 | 4.938e-03 | 5.175e-03 | 5.958e-03 | 0.0% | 9.589e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | Torch-Solve | 6.368e-01 | 6.338 | 1.212e-04 | 1.376e-04 | 1.213e-04 | 0.0% | 6.373e-01 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 0.000e+00 | 7.376 | 1.186e-02 | 5.537e+00 | 1.276e-02 | 100.0% | 2.725e-01 |
| spd | 1 | 256 | 256 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.997e+01 | 1.752 | 1.642e-07 | 1.201e-04 | 1.700e-07 | 0.0% | 2.068e+01 |
| spd | 1 | 256 | 256 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 2.153e+01 | 1.953 | 1.647e-07 | 1.559e-04 | 1.715e-07 | 0.0% | 2.242e+01 |
| spd | 1 | 512 | 512 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 9.642e+00 | 2.291 | 1.663e-07 | 7.764e-05 | 1.730e-07 | 0.0% | 1.003e+01 |
| spd | 1 | 512 | 512 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.089e+01 | 2.881 | 1.635e-07 | 1.156e-04 | 1.665e-07 | 0.0% | 1.109e+01 |
| spd | 1 | 1024 | 1 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 2.715e+01 | 2.110 | 3.513e-07 | 4.312e-05 | 3.636e-07 | 0.0% | 2.810e+01 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.264e+01 | 1.889 | 3.573e-07 | 8.167e-05 | 3.660e-07 | 0.0% | 3.343e+01 |
| spd | 1 | 1024 | 16 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.214e+01 | 2.078 | 1.676e-07 | 4.333e-05 | 1.723e-07 | 0.0% | 1.248e+01 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.223e+01 | 1.840 | 1.679e-07 | 8.223e-05 | 1.718e-07 | 0.0% | 1.251e+01 |
| spd | 1 | 1024 | 64 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 1.153e+01 | 2.016 | 1.677e-07 | 4.339e-05 | 1.710e-07 | 0.0% | 1.176e+01 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 1.153e+01 | 1.995 | 1.681e-07 | 8.176e-05 | 1.713e-07 | 0.0% | 1.175e+01 |
| spd | 1 | 1024 | 1024 | gaussian_spd | Torch-Cholesky-Solve-ReuseFactor | 3.200e+00 | 3.468 | 1.891e-07 | 4.285e-05 | 1.914e-07 | 0.0% | 3.239e+00 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | Torch-Cholesky-Solve-ReuseFactor | 3.215e+00 | 3.580 | 1.870e-07 | 8.165e-05 | 1.898e-07 | 0.0% | 3.263e+00 |
| spd | 2 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.464e+00 | 1.861 | 2.828e-03 | 2.834e-03 | 2.934e-03 | 0.0% | 4.631e+00 |
| spd | 2 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.286e+00 | 2.195 | 2.682e-03 | 2.686e-03 | 2.717e-03 | 0.0% | 6.368e+00 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.517e+00 | 3.732 | 2.936e-03 | 2.942e-03 | 2.948e-03 | 0.0% | 2.527e+00 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.247e+00 | 2.723 | 2.878e-03 | 2.881e-03 | 3.158e-03 | 0.0% | 2.466e+00 |
| spd | 2 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.663e+00 | 2.052 | 2.657e-03 | 2.659e-03 | 2.822e-03 | 0.0% | 2.828e+00 |
| spd | 2 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.031 | 2.136e-02 | nan | nan | nan% | nan |
| spd | 2 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 1.611e+00 | 3.350 | 2.712e-03 | 2.714e-03 | 2.858e-03 | 0.0% | 1.698e+00 |
| spd | 2 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 1.550e+00 | 2.994 | 2.660e-03 | 2.662e-03 | 2.996e-03 | 0.0% | 1.746e+00 |
| spd | 2 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.888 | 2.185e-02 | nan | nan | nan% | nan |
| spd | 2 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.658e+00 | 2.305 | 2.674e-03 | 2.675e-03 | 2.705e-03 | 0.0% | 2.689e+00 |
| spd | 2 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 1.700e+00 | 2.990 | 2.714e-03 | 2.715e-03 | 2.973e-03 | 0.0% | 1.862e+00 |
| spd | 2 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 1.940 | 2.185e-02 | nan | nan | nan% | nan |
| spd | 2 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 1.737e+00 | 4.740 | 2.654e-03 | 2.655e-03 | 2.949e-03 | 0.0% | 1.930e+00 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 7.263e-01 | 3.609 | 4.551e-03 | 4.557e-03 | 6.256e-03 | 0.0% | 9.984e-01 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 5.313e-01 | 5.620 | 7.075e-03 | 7.079e-03 | 7.079e-03 | 0.0% | 5.316e-01 |
| spd | 4 | 256 | 256 | gaussian_spd | Chebyshev-Apply | 4.939e+00 | 2.024 | 1.901e-03 | 1.902e-03 | 1.907e-03 | 0.0% | 4.955e+00 |
| spd | 4 | 256 | 256 | illcond_1e6 | Chebyshev-Apply | 6.695e+00 | 1.792 | 1.901e-03 | 1.901e-03 | 1.917e-03 | 0.0% | 6.751e+00 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.023e+00 | 2.737 | 2.929e-03 | 2.929e-03 | 3.630e-03 | 0.0% | 2.507e+00 |
| spd | 4 | 512 | 512 | illcond_1e6 | Chebyshev-Apply | 2.428e+00 | 2.425 | 1.896e-03 | 1.896e-03 | 1.900e-03 | 0.0% | 2.433e+00 |
| spd | 4 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 2.796e+00 | 2.315 | 1.856e-03 | 1.856e-03 | 1.949e-03 | 0.0% | 2.936e+00 |
| spd | 4 | 1024 | 1 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.653 | 1.123e-02 | nan | nan | nan% | nan |
| spd | 4 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 3.032e+00 | 2.253 | 1.907e-03 | 1.908e-03 | 1.963e-03 | 0.0% | 3.121e+00 |
| spd | 4 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 2.082e+00 | 2.928 | 1.900e-03 | 1.900e-03 | 1.905e-03 | 0.0% | 2.087e+00 |
| spd | 4 | 1024 | 16 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.397 | 1.172e-02 | nan | nan | nan% | nan |
| spd | 4 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 2.613e+00 | 2.851 | 1.908e-03 | 1.909e-03 | 1.916e-03 | 0.0% | 2.624e+00 |
| spd | 4 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 2.515e+00 | 2.526 | 1.904e-03 | 1.904e-03 | 1.918e-03 | 0.0% | 2.533e+00 |
| spd | 4 | 1024 | 64 | gram_rhs_gtb | PE-Quad-Coupled-Apply-Primal-Gram | 0.000e+00 | 2.410 | 1.154e-02 | nan | nan | nan% | nan |
| spd | 4 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 2.271e+00 | 2.903 | 1.904e-03 | 1.904e-03 | 1.918e-03 | 0.0% | 2.288e+00 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 8.674e-01 | 4.727 | 3.649e-03 | 3.651e-03 | 3.736e-03 | 0.0% | 8.881e-01 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 5.384e-01 | 6.403 | 3.796e-03 | 3.801e-03 | 3.803e-03 | 0.0% | 5.394e-01 |
