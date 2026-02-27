# Solver Benchmark A/B Report

Generated: 2026-02-27T18:14:19

A: baseline
B: adaptive_main

| kind | p | n | k | case | method | baseline_total_ms | adaptive_main_total_ms | delta_ms(B-A) | delta_pct | baseline_iter_ms | adaptive_main_iter_ms | baseline_relerr | adaptive_main_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.857 | 3.168 | 1.311 | 70.60% | 1.698 | 2.980 | 5.646e-03 | 5.646e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.963 | 2.842 | 0.879 | 44.78% | 1.755 | 2.690 | 4.456e-03 | 4.456e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.686 | 2.746 | 1.060 | 62.87% | 1.498 | 2.576 | 7.538e-03 | 7.538e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.135 | 5.433 | 0.298 | 5.80% | 4.916 | 5.272 | 1.968e-03 | 1.968e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.560 | 2.841 | 1.281 | 82.12% | 1.398 | 2.692 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.573 | 2.752 | 1.179 | 74.95% | 1.423 | 2.604 | 5.676e-03 | 5.676e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.605 | 2.894 | 1.289 | 80.31% | 1.454 | 2.746 | 7.507e-03 | 7.507e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.389 | 5.723 | 0.334 | 6.20% | 5.235 | 5.554 | 9.033e-03 | 9.033e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.608 | 2.742 | 1.134 | 70.52% | 1.461 | 2.593 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.635 | 2.781 | 1.146 | 70.09% | 1.428 | 2.622 | 6.317e-03 | 6.317e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.669 | 2.926 | 1.257 | 75.31% | 1.456 | 2.713 | 7.568e-03 | 7.568e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.315 | 5.687 | 0.372 | 7.00% | 5.161 | 5.540 | 1.031e-02 | 1.031e-02 | 1.000 |
