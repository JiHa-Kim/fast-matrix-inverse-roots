# Solver Benchmark A/B Report

Generated: 2026-02-27T23:23:14

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

A: early_diag
B: early_fro

| kind | p | n | k | case | method | early_diag_total_ms | early_fro_total_ms | delta_ms(B-A) | delta_pct | early_diag_iter_ms | early_fro_iter_ms | early_diag_relerr | early_fro_relerr | relerr_ratio(B/A) | early_diag_relerr_p90 | early_fro_relerr_p90 | early_diag_fail_rate | early_fro_fail_rate | early_diag_q_per_ms | early_fro_q_per_ms | q_per_ms_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.289 | 2.207 | -0.082 | -3.58% | 1.972 | 1.967 | 5.517e-03 | 5.517e-03 | 1.000 | 5.688e-03 | 5.688e-03 | 0.0% | 0.0% | 1.145e+00 | 1.148e+00 | 1.003 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 2.300 | 2.241 | -0.059 | -2.57% | 2.141 | 2.084 | 4.141e-03 | 4.141e-03 | 1.000 | 5.542e-03 | 5.542e-03 | 0.0% | 0.0% | 1.113e+00 | 1.144e+00 | 1.028 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.909 | 2.911 | 1.002 | 52.49% | 1.721 | 2.569 | 7.409e-03 | 7.409e-03 | 1.000 | 7.684e-03 | 7.684e-03 | 0.0% | 0.0% | 1.238e+00 | 8.292e-01 | 0.670 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.089 | 5.482 | 0.393 | 7.72% | 4.890 | 5.250 | 1.861e-03 | 1.861e-03 | 1.000 | 5.153e-03 | 5.153e-03 | 0.0% | 0.0% | 5.584e-01 | 5.200e-01 | 0.931 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.676 | 2.192 | 0.516 | 30.79% | 1.533 | 2.041 | 5.595e-03 | 5.595e-03 | 1.000 | 5.684e-03 | 5.684e-03 | 0.0% | 0.0% | 1.469e+00 | 1.103e+00 | 0.751 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.583 | 2.237 | 0.654 | 41.31% | 1.410 | 1.799 | 5.409e-03 | 5.409e-03 | 1.000 | 6.164e-03 | 6.164e-03 | 0.0% | 0.0% | 1.608e+00 | 1.260e+00 | 0.784 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.732 | 1.885 | 0.153 | 8.83% | 1.566 | 1.629 | 7.319e-03 | 7.319e-03 | 1.000 | 7.810e-03 | 7.810e-03 | 0.0% | 0.0% | 1.363e+00 | 1.311e+00 | 0.962 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.714 | 5.742 | 0.028 | 0.49% | 5.477 | 5.536 | 8.857e-03 | 8.857e-03 | 1.000 | 1.105e-02 | 1.105e-02 | 0.0% | 0.0% | 3.748e-01 | 3.708e-01 | 0.989 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.418 | 2.008 | -0.410 | -16.96% | 1.869 | 1.855 | 5.579e-03 | 5.579e-03 | 1.000 | 5.768e-03 | 5.768e-03 | 0.0% | 0.0% | 1.206e+00 | 1.215e+00 | 1.007 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 2.160 | 2.268 | 0.108 | 5.00% | 1.963 | 2.018 | 6.100e-03 | 6.100e-03 | 1.000 | 6.145e-03 | 6.145e-03 | 0.0% | 0.0% | 1.128e+00 | 1.098e+00 | 0.973 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.736 | 2.048 | 0.312 | 17.97% | 1.547 | 1.844 | 7.362e-03 | 7.362e-03 | 1.000 | 7.610e-03 | 7.610e-03 | 0.0% | 0.0% | 1.378e+00 | 1.157e+00 | 0.840 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.482 | 5.716 | 0.234 | 4.27% | 5.327 | 5.559 | 1.017e-02 | 1.017e-02 | 1.000 | 1.235e-02 | 1.235e-02 | 0.0% | 0.0% | 3.740e-01 | 3.584e-01 | 0.958 |

## A/B Summary

| metric | count | share |
|---|---:|---:|
| B faster (total_ms) | 3 / 12 | 25.0% |
| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | 12 / 12 | 100.0% |
| B better assessment score | 3 / 12 | 25.0% |
