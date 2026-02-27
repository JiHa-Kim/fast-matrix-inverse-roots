# Solver Benchmark A/B Report

Generated: 2026-02-27T18:00:26

A: baseline
B: monotone

| kind | p | n | k | case | method | baseline_total_ms | monotone_total_ms | delta_ms(B-A) | delta_pct | baseline_iter_ms | monotone_iter_ms | baseline_relerr | monotone_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.943 | 3.236 | 1.293 | 66.55% | 1.718 | 3.055 | 5.646e-03 | 5.646e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.916 | 3.247 | 1.331 | 69.47% | 1.691 | 3.068 | 4.456e-03 | 4.456e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.590 | 3.104 | 1.514 | 95.22% | 1.434 | 2.918 | 7.538e-03 | 7.538e-03 | 1.000 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.076 | 5.739 | 0.663 | 13.06% | 4.926 | 5.588 | 1.968e-03 | 1.968e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.647 | 2.885 | 1.238 | 75.17% | 1.451 | 2.734 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.597 | 2.918 | 1.321 | 82.72% | 1.436 | 2.754 | 5.676e-03 | 5.676e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.685 | 2.873 | 1.188 | 70.50% | 1.520 | 2.696 | 7.507e-03 | 7.507e-03 | 1.000 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.396 | 5.904 | 0.508 | 9.41% | 5.249 | 5.750 | 9.033e-03 | 9.033e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.788 | 3.048 | 1.260 | 70.47% | 1.457 | 2.884 | 5.829e-03 | 5.829e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.611 | 3.059 | 1.448 | 89.88% | 1.425 | 2.890 | 6.317e-03 | 6.317e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.753 | 4.491 | 2.738 | 156.19% | 1.606 | 4.286 | 7.568e-03 | 7.568e-03 | 1.000 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.426 | 6.415 | 0.989 | 18.23% | 5.257 | 6.219 | 1.031e-02 | 1.031e-02 | 1.000 |
