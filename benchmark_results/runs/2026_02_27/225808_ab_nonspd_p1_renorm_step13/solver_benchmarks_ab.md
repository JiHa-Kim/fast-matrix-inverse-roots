# Solver Benchmark A/B Report

Generated: 2026-02-27T22:58:31

Assessment metrics:
- `relerr`: median relative error across trials.
- `relerr_p90`: 90th percentile relative error (tail quality).
- `fail_rate`: fraction of failed/non-finite trials.
- `q_per_ms`: `max(0, -log10(relerr)) / iter_ms`.
- assessment score: `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.

A: renorm_off
B: renorm_on

| kind | p | n | k | case | method | renorm_off_total_ms | renorm_on_total_ms | delta_ms(B-A) | delta_pct | renorm_off_iter_ms | renorm_on_iter_ms | renorm_off_relerr | renorm_on_relerr | relerr_ratio(B/A) | renorm_off_relerr_p90 | renorm_on_relerr_p90 | renorm_off_fail_rate | renorm_on_fail_rate | renorm_off_q_per_ms | renorm_on_q_per_ms | q_per_ms_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.218 | 2.849 | 0.631 | 28.45% | 2.059 | 2.642 | 5.517e-03 | 7.355e-03 | 1.333 | 5.688e-03 | 7.849e-03 | 0.0% | 0.0% | 1.097e+00 | 8.075e-01 | 0.736 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 2.673 | 3.034 | 0.361 | 13.51% | 2.455 | 2.839 | 4.141e-03 | 6.064e-03 | 1.464 | 5.542e-03 | 6.492e-03 | 0.0% | 0.0% | 9.708e-01 | 7.810e-01 | 0.804 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 2.088 | 7.080 | 4.992 | 239.08% | 1.879 | 6.837 | 7.409e-03 | 1.684e-03 | 0.227 | 7.684e-03 | 1.719e-03 | 0.0% | 0.0% | 1.133e+00 | 4.056e-01 | 0.358 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.166 | 5.631 | 0.465 | 9.00% | 4.997 | 5.406 | 1.861e-03 | 1.861e-03 | 1.000 | 5.153e-03 | 5.153e-03 | 0.0% | 0.0% | 5.464e-01 | 5.051e-01 | 0.924 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.847 | 2.730 | 0.883 | 47.81% | 1.622 | 2.577 | 5.595e-03 | 7.121e-03 | 1.273 | 5.684e-03 | 7.797e-03 | 0.0% | 0.0% | 1.388e+00 | 8.333e-01 | 0.600 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.902 | 2.828 | 0.926 | 48.69% | 1.633 | 2.649 | 5.409e-03 | 6.503e-03 | 1.202 | 6.164e-03 | 7.100e-03 | 0.0% | 0.0% | 1.388e+00 | 8.257e-01 | 0.595 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.901 | 7.713 | 5.812 | 305.73% | 1.712 | 7.533 | 7.319e-03 | 1.654e-03 | 0.226 | 7.810e-03 | 1.687e-03 | 0.0% | 0.0% | 1.248e+00 | 3.692e-01 | 0.296 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.482 | 6.062 | 0.580 | 10.58% | 5.308 | 5.860 | 8.857e-03 | 8.857e-03 | 1.000 | 1.105e-02 | 1.105e-02 | 0.0% | 0.0% | 3.868e-01 | 3.503e-01 | 0.906 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.769 | 3.478 | 1.709 | 96.61% | 1.619 | 3.028 | 5.579e-03 | 6.372e-03 | 1.142 | 5.768e-03 | 7.797e-03 | 0.0% | 0.0% | 1.392e+00 | 7.251e-01 | 0.521 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.744 | 3.019 | 1.275 | 73.11% | 1.569 | 2.648 | 6.100e-03 | 7.149e-03 | 1.172 | 6.145e-03 | 7.776e-03 | 0.0% | 0.0% | 1.411e+00 | 8.104e-01 | 0.574 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 2.141 | 8.065 | 5.924 | 276.69% | 1.965 | 7.912 | 7.362e-03 | 1.660e-03 | 0.225 | 7.610e-03 | 1.672e-03 | 0.0% | 0.0% | 1.086e+00 | 3.514e-01 | 0.324 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.671 | 6.116 | 0.445 | 7.85% | 5.519 | 5.931 | 1.017e-02 | 1.017e-02 | 1.000 | 1.235e-02 | 1.235e-02 | 0.0% | 0.0% | 3.610e-01 | 3.360e-01 | 0.931 |

## A/B Summary

| metric | count | share |
|---|---:|---:|
| B faster (total_ms) | 0 / 12 | 0.0% |
| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | 6 / 12 | 50.0% |
| B better assessment score | 1 / 12 | 8.3% |
