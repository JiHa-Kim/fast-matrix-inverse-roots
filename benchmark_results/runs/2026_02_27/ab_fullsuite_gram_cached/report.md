# Solver Benchmark A/B Report

Generated: 2026-02-27T17:19:26

A: baseline
B: gram_cached

| kind | p | n | k | case | method | baseline_total_ms | gram_cached_total_ms | delta_ms(B-A) | delta_pct | baseline_iter_ms | gram_cached_iter_ms | baseline_relerr | gram_cached_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.462 | 2.089 | -0.373 | -15.15% | 0.660 | 0.660 | 3.799e-03 | 3.799e-03 | 1.000 |
| spd | 2 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.942 | 1.941 | -0.001 | -0.05% | 0.661 | 0.807 | 3.311e-03 | 3.311e-03 | 1.000 |
| spd | 2 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.100 | 2.187 | 0.087 | 4.14% | 0.769 | 0.712 | 3.204e-03 | 3.204e-03 | 1.000 |
| spd | 2 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.155 | 2.102 | -0.053 | -2.46% | 0.789 | 0.860 | 3.555e-03 | 3.555e-03 | 1.000 |
| spd | 2 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 2.799 | 2.869 | 0.070 | 2.50% | 1.605 | 1.582 | 3.769e-03 | 3.769e-03 | 1.000 |
| spd | 2 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.948 | 2.504 | -0.444 | -15.06% | 1.606 | 1.311 | 5.005e-03 | 5.005e-03 | 1.000 |
| spd | 2 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 3.249 | 2.608 | -0.641 | -19.73% | 1.610 | 1.306 | 3.937e-03 | 3.937e-03 | 1.000 |
| spd | 2 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.136 | 4.416 | 1.280 | 40.82% | 1.326 | 1.621 | 4.944e-03 | 4.944e-03 | 1.000 |
| spd | 2 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 3.210 | 3.167 | -0.043 | -1.34% | 1.366 | 1.744 | 3.860e-03 | 3.860e-03 | 1.000 |
| spd | 2 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.359 | 3.828 | 0.469 | 13.96% | 1.622 | 1.914 | 4.974e-03 | 4.974e-03 | 1.000 |
| spd | 2 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.574 | 3.542 | -0.032 | -0.90% | 2.363 | 2.355 | 4.578e-03 | 4.578e-03 | 1.000 |
| spd | 2 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.247 | 3.435 | 0.188 | 5.79% | 1.908 | 1.968 | 7.294e-03 | 7.294e-03 | 1.000 |
| spd | 4 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.701 | 2.509 | -0.192 | -7.11% | 0.719 | 0.785 | 3.693e-03 | 3.693e-03 | 1.000 |
| spd | 4 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.953 | 2.509 | 0.556 | 28.47% | 0.846 | 0.984 | 4.120e-03 | 4.120e-03 | 1.000 |
| spd | 4 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 2.196 | 2.783 | 0.587 | 26.73% | 0.846 | 0.795 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.046 | 2.142 | 0.096 | 4.69% | 0.897 | 0.981 | 3.418e-03 | 3.418e-03 | 1.000 |
| spd | 4 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 3.533 | 3.288 | -0.245 | -6.93% | 1.970 | 1.984 | 3.723e-03 | 3.723e-03 | 1.000 |
| spd | 4 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.890 | 3.072 | 0.182 | 6.30% | 1.642 | 1.897 | 3.479e-03 | 3.479e-03 | 1.000 |
| spd | 4 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 2.908 | 3.293 | 0.385 | 13.24% | 1.655 | 1.885 | 3.601e-03 | 3.601e-03 | 1.000 |
| spd | 4 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.219 | 3.174 | -0.045 | -1.40% | 2.021 | 1.705 | 3.464e-03 | 3.464e-03 | 1.000 |
| spd | 4 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 4.608 | 2.933 | -1.675 | -36.35% | 2.551 | 1.700 | 3.647e-03 | 3.647e-03 | 1.000 |
| spd | 4 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.223 | 3.575 | 0.352 | 10.92% | 1.693 | 1.687 | 3.464e-03 | 3.464e-03 | 1.000 |
| spd | 4 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 4.602 | 4.738 | 0.136 | 2.96% | 2.736 | 2.740 | 3.998e-03 | 3.998e-03 | 1.000 |
| spd | 4 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.582 | 3.385 | -0.197 | -5.50% | 2.289 | 2.295 | 4.181e-03 | 4.181e-03 | 1.000 |
