# Production Benchmark Results (Readable Summary)

Generated from run artifact: `benchmark_results/runs/2026_02_27/233459_production_fullsuite_step15/solver_benchmarks.md`
Generated at: `2026-02-27T23:41:01`

## Run Flags

- `dtype`: `bf16`
- `extra_args`: ``
- `only`: ``
- `timing_reps`: `10`
- `timing_warmup_reps`: `2`
- `trials`: `10`

## SPD Solve (p=1)

| p | n | k | case | best method | total ms | relerr | relerr_p90 | score |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 256 | 256 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 1.607 | 1.662e-03 | 1.666e-03 | 1.623e+01 |
| 1 | 256 | 256 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 1.737 | 1.660e-03 | 1.664e-03 | 1.335e+01 |
| 1 | 512 | 512 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 2.221 | 1.662e-03 | 1.663e-03 | 4.700e+00 |
| 1 | 512 | 512 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 2.061 | 1.662e-03 | 1.664e-03 | 4.758e+00 |
| 1 | 1024 | 1 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 2.050 | 1.654e-03 | 1.701e-03 | 1.411e+01 |
| 1 | 1024 | 1 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 1.437 | 1.672e-03 | 1.713e-03 | 1.393e+01 |
| 1 | 1024 | 16 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 1.828 | 1.666e-03 | 1.673e-03 | 5.142e+00 |
| 1 | 1024 | 16 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 1.978 | 1.667e-03 | 1.679e-03 | 5.140e+00 |
| 1 | 1024 | 64 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 1.701 | 1.658e-03 | 1.664e-03 | 4.783e+00 |
| 1 | 1024 | 64 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 2.505 | 1.662e-03 | 1.666e-03 | 4.800e+00 |
| 1 | 1024 | 1024 | gaussian_spd | **Torch-Cholesky-Solve-ReuseFactor** | 4.329 | 1.660e-03 | 1.662e-03 | 1.293e+00 |
| 1 | 1024 | 1024 | illcond_1e6 | **Torch-Cholesky-Solve-ReuseFactor** | 3.786 | 1.660e-03 | 1.661e-03 | 1.295e+00 |

## SPD Inverse Square Root (p=2)

| p | n | k | case | best method | total ms | relerr | relerr_p90 | score |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 2 | 256 | 256 | gaussian_spd | **Chebyshev-Apply** | 1.792 | 2.828e-03 | 2.934e-03 | 4.477e+00 |
| 2 | 256 | 256 | illcond_1e6 | **Chebyshev-Apply** | 2.008 | 2.682e-03 | 2.717e-03 | 6.313e+00 |
| 2 | 512 | 512 | gaussian_spd | **PE-Quad-Coupled-Apply** | 2.177 | 2.936e-03 | 2.948e-03 | 2.500e+00 |
| 2 | 512 | 512 | illcond_1e6 | **PE-Quad-Coupled-Apply** | 2.691 | 2.878e-03 | 3.158e-03 | 2.128e+00 |
| 2 | 1024 | 1 | gaussian_spd | **Chebyshev-Apply** | 1.948 | 2.657e-03 | 2.822e-03 | 5.494e+00 |
| 2 | 1024 | 1 | illcond_1e6 | **Chebyshev-Apply** | 2.102 | 2.712e-03 | 2.858e-03 | 5.509e+00 |
| 2 | 1024 | 16 | gaussian_spd | **Chebyshev-Apply** | 1.682 | 2.660e-03 | 2.996e-03 | 4.596e+00 |
| 2 | 1024 | 16 | illcond_1e6 | **Chebyshev-Apply** | 1.930 | 2.674e-03 | 2.705e-03 | 5.133e+00 |
| 2 | 1024 | 64 | gaussian_spd | **Chebyshev-Apply** | 1.955 | 2.714e-03 | 2.973e-03 | 4.329e+00 |
| 2 | 1024 | 64 | illcond_1e6 | **Chebyshev-Apply** | 1.931 | 2.654e-03 | 2.949e-03 | 4.279e+00 |
| 2 | 1024 | 1024 | gaussian_spd | **PE-Quad-Coupled-Apply** | 3.761 | 4.551e-03 | 6.256e-03 | 7.275e-01 |
| 2 | 1024 | 1024 | illcond_1e6 | **PE-Quad-Coupled-Apply** | 4.084 | 7.075e-03 | 7.079e-03 | 1.065e+00 |

## SPD Inverse Fourth Root (p=4)

| p | n | k | case | best method | total ms | relerr | relerr_p90 | score |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 4 | 256 | 256 | gaussian_spd | **Chebyshev-Apply** | 2.319 | 1.901e-03 | 1.907e-03 | 4.942e+00 |
| 4 | 256 | 256 | illcond_1e6 | **Chebyshev-Apply** | 2.219 | 1.901e-03 | 1.917e-03 | 6.709e+00 |
| 4 | 512 | 512 | gaussian_spd | **PE-Quad-Coupled-Apply** | 2.517 | 2.929e-03 | 3.630e-03 | 2.183e+00 |
| 4 | 512 | 512 | illcond_1e6 | **PE-Quad-Coupled-Apply** | 2.911 | 3.610e-03 | 3.657e-03 | 2.613e+00 |
| 4 | 1024 | 1 | gaussian_spd | **Chebyshev-Apply** | 1.955 | 1.856e-03 | 1.949e-03 | 5.887e+00 |
| 4 | 1024 | 1 | illcond_1e6 | **Chebyshev-Apply** | 1.568 | 1.907e-03 | 1.963e-03 | 5.976e+00 |
| 4 | 1024 | 16 | gaussian_spd | **Chebyshev-Apply** | 1.599 | 1.900e-03 | 1.905e-03 | 5.490e+00 |
| 4 | 1024 | 16 | illcond_1e6 | **Chebyshev-Apply** | 2.351 | 1.908e-03 | 1.916e-03 | 5.456e+00 |
| 4 | 1024 | 64 | gaussian_spd | **Chebyshev-Apply** | 1.826 | 1.904e-03 | 1.918e-03 | 4.979e+00 |
| 4 | 1024 | 64 | illcond_1e6 | **Chebyshev-Apply** | 2.371 | 1.904e-03 | 1.918e-03 | 4.988e+00 |
| 4 | 1024 | 1024 | gaussian_spd | **PE-Quad-Coupled-Apply** | 3.898 | 3.649e-03 | 3.736e-03 | 9.875e-01 |
| 4 | 1024 | 1024 | illcond_1e6 | **PE-Quad-Coupled-Apply** | 3.628 | 3.796e-03 | 3.803e-03 | 1.019e+00 |

## Non-SPD Solve (p=1)

| p | n | k | case | best method | total ms | relerr | relerr_p90 | score |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 256 | 256 | gaussian_shifted | **Torch-Solve** | 1.119 | 1.653e-03 | 1.661e-03 | 2.934e+00 |
| 1 | 256 | 256 | nonnormal_upper | **Torch-Solve** | 1.053 | 1.658e-03 | 1.663e-03 | 3.312e+00 |
| 1 | 256 | 256 | similarity_posspec | **Torch-Solve** | 1.188 | 1.661e-03 | 1.668e-03 | 2.802e+00 |
| 1 | 256 | 256 | similarity_posspec_hard | **Torch-Solve** | 0.986 | 1.664e-03 | 1.668e-03 | 3.274e+00 |
| 1 | 512 | 512 | gaussian_shifted | **PE-Quad-Coupled-Apply** | 1.552 | 4.385e-03 | 4.520e-03 | 1.701e+00 |
| 1 | 512 | 512 | nonnormal_upper | **PE-Quad-Coupled-Apply** | 1.667 | 4.788e-03 | 6.180e-03 | 1.334e+00 |
| 1 | 512 | 512 | similarity_posspec | **PE-Quad-Coupled-Apply** | 1.447 | 4.899e-03 | 5.057e-03 | 1.771e+00 |
| 1 | 512 | 512 | similarity_posspec_hard | **Torch-Solve** | 3.011 | 4.714e-03 | 5.204e-03 | 7.413e-01 |
| 1 | 1024 | 1 | gaussian_shifted | **PE-Quad-Coupled-Apply** | 1.790 | 5.730e-03 | 5.981e-03 | 1.334e+00 |
| 1 | 1024 | 1 | nonnormal_upper | **PE-Quad-Coupled-Apply** | 1.699 | 5.012e-03 | 5.491e-03 | 1.360e+00 |
| 1 | 1024 | 1 | similarity_posspec | **Torch-Solve** | 4.480 | 1.673e-03 | 1.704e-03 | 6.357e-01 |
| 1 | 1024 | 1 | similarity_posspec_hard | **Torch-Solve** | 4.388 | 1.857e-03 | 3.921e-03 | 3.052e-01 |
| 1 | 1024 | 16 | gaussian_shifted | **PE-Quad-Coupled-Apply** | 1.594 | 5.627e-03 | 5.745e-03 | 1.532e+00 |
| 1 | 1024 | 16 | nonnormal_upper | **PE-Quad-Coupled-Apply** | 1.688 | 4.138e-03 | 6.058e-03 | 1.098e+00 |
| 1 | 1024 | 16 | similarity_posspec | **PE-Quad-Coupled-Apply** | 1.870 | 7.364e-03 | 7.575e-03 | 1.239e+00 |
| 1 | 1024 | 16 | similarity_posspec_hard | **Torch-Solve** | 4.861 | 7.698e-03 | 3.300e-02 | 1.062e-01 |
| 1 | 1024 | 64 | gaussian_shifted | **PE-Quad-Coupled-Apply** | 1.678 | 5.667e-03 | 5.720e-03 | 1.472e+00 |
| 1 | 1024 | 64 | nonnormal_upper | **PE-Quad-Coupled-Apply** | 1.756 | 4.538e-03 | 6.129e-03 | 1.082e+00 |
| 1 | 1024 | 64 | similarity_posspec | **PE-Quad-Coupled-Apply** | 1.717 | 7.504e-03 | 7.661e-03 | 1.349e+00 |
| 1 | 1024 | 64 | similarity_posspec_hard | **Torch-Solve** | 4.873 | 1.157e-02 | 3.692e-02 | 1.290e-01 |
| 1 | 1024 | 1024 | gaussian_shifted | **PE-Quad-Coupled-Apply** | 3.141 | 5.497e-03 | 5.540e-03 | 7.521e-01 |
| 1 | 1024 | 1024 | nonnormal_upper | **PE-Quad-Coupled-Apply** | 2.653 | 4.938e-03 | 5.958e-03 | 7.692e-01 |
| 1 | 1024 | 1024 | similarity_posspec | **Torch-Solve** | 6.514 | 1.665e-03 | 1.667e-03 | 4.549e-01 |
| 1 | 1024 | 1024 | similarity_posspec_hard | **Torch-Solve** | 6.309 | 1.186e-02 | 1.276e-02 | 2.918e-01 |

## Gram-RHS Speed Path

| p | n | k | case | best method | total ms | relerr | relerr_p90 | score |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 2 | 1024 | 1 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 2.337 | 2.136e-02 | nan | 0.000e+00 |
| 2 | 1024 | 16 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 1.913 | 2.185e-02 | nan | 0.000e+00 |
| 2 | 1024 | 64 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 1.910 | 2.185e-02 | nan | 0.000e+00 |
| 4 | 1024 | 1 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 2.405 | 1.123e-02 | nan | 0.000e+00 |
| 4 | 1024 | 16 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 2.389 | 1.172e-02 | nan | 0.000e+00 |
| 4 | 1024 | 64 | gram_rhs_gtb | **PE-Quad-Coupled-Apply-Primal-Gram** | 2.393 | 1.154e-02 | nan | 0.000e+00 |

## Raw Full Report

For the full per-method table (all methods per cell), see:
- `benchmark_results/runs/2026_02_27/233459_production_fullsuite_step15/solver_benchmarks.md`
- `benchmark_results/runs/2026_02_27/233459_production_fullsuite_step15/run_manifest.json`
