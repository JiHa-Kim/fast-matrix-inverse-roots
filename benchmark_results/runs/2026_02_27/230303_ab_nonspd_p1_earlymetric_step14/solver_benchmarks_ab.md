# Solver Benchmark A/B Report

Generated: 2026-02-27T23:03:25

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

A: early_diag
B: early_fro

| kind | p | n | k | case | method | early_diag_total_ms | early_fro_total_ms | delta_ms(B-A) | delta_pct | early_diag_iter_ms | early_fro_iter_ms | early_diag_relerr | early_fro_relerr | relerr_ratio(B/A) | early_diag_relerr_p90 | early_fro_relerr_p90 | early_diag_fail_rate | early_fro_fail_rate | early_diag_q_per_ms | early_fro_q_per_ms | q_per_ms_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.057 | 2.065 | 0.008 | 0.39% | 1.879 | 1.827 | 5.517e-03 | 5.517e-03 | 1.000 | 5.688e-03 | 5.688e-03 | 0.0% | 0.0% | 1.202e+00 | 1.236e+00 | 1.028 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.922 | 2.116 | 0.194 | 10.09% | 1.764 | 1.909 | 4.141e-03 | 4.141e-03 | 1.000 | 5.542e-03 | 5.542e-03 | 0.0% | 0.0% | 1.351e+00 | 1.249e+00 | 0.925 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.984 | 2.131 | 0.147 | 7.41% | 1.793 | 1.700 | 7.409e-03 | 7.409e-03 | 1.000 | 7.684e-03 | 7.684e-03 | 0.0% | 0.0% | 1.188e+00 | 1.253e+00 | 1.055 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.505 | 5.467 | -0.038 | -0.69% | 5.313 | 5.188 | 1.861e-03 | 1.861e-03 | 1.000 | 5.153e-03 | 5.153e-03 | 0.0% | 0.0% | 5.139e-01 | 5.263e-01 | 1.024 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.678 | 1.758 | 0.080 | 4.77% | 1.483 | 1.598 | 5.595e-03 | 5.595e-03 | 1.000 | 5.684e-03 | 5.684e-03 | 0.0% | 0.0% | 1.518e+00 | 1.409e+00 | 0.928 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.621 | 2.155 | 0.534 | 32.94% | 1.443 | 1.682 | 5.409e-03 | 5.409e-03 | 1.000 | 6.164e-03 | 6.164e-03 | 0.0% | 0.0% | 1.571e+00 | 1.347e+00 | 0.857 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 2.430 | 2.641 | 0.211 | 8.68% | 2.145 | 2.456 | 7.319e-03 | 7.319e-03 | 1.000 | 7.810e-03 | 7.810e-03 | 0.0% | 0.0% | 9.956e-01 | 8.697e-01 | 0.874 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.612 | 5.561 | -0.051 | -0.91% | 5.309 | 5.349 | 8.857e-03 | 8.857e-03 | 1.000 | 1.105e-02 | 1.105e-02 | 0.0% | 0.0% | 3.866e-01 | 3.837e-01 | 0.992 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.818 | 1.694 | -0.124 | -6.82% | 1.662 | 1.523 | 5.579e-03 | 5.579e-03 | 1.000 | 5.768e-03 | 5.768e-03 | 0.0% | 0.0% | 1.356e+00 | 1.480e+00 | 1.091 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.850 | 2.255 | 0.405 | 21.89% | 1.473 | 1.839 | 6.100e-03 | 6.100e-03 | 1.000 | 6.145e-03 | 6.145e-03 | 0.0% | 0.0% | 1.504e+00 | 1.204e+00 | 0.801 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 2.012 | 1.814 | -0.198 | -9.84% | 1.819 | 1.640 | 7.362e-03 | 7.362e-03 | 1.000 | 7.610e-03 | 7.610e-03 | 0.0% | 0.0% | 1.173e+00 | 1.301e+00 | 1.109 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.748 | 6.033 | 0.285 | 4.96% | 5.388 | 5.844 | 1.017e-02 | 1.017e-02 | 1.000 | 1.235e-02 | 1.235e-02 | 0.0% | 0.0% | 3.698e-01 | 3.409e-01 | 0.922 |

## A/B Summary

| metric | count | share |
|---|---:|---:|
| B faster (total_ms) | 4 / 12 | 33.3% |
| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | 12 / 12 | 100.0% |
| B better assessment score | 5 / 12 | 41.7% |
