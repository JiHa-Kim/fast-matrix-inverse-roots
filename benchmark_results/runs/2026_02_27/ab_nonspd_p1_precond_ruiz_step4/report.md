# Solver Benchmark A/B Report

Generated: 2026-02-27T18:12:54

A: row_norm
B: ruiz

| kind | p | n | k | case | method | row_norm_total_ms | ruiz_total_ms | delta_ms(B-A) | delta_pct | row_norm_iter_ms | ruiz_iter_ms | row_norm_relerr | ruiz_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.948 | 2.197 | 0.249 | 12.78% | 1.781 | 1.760 | 5.646e-03 | 5.859e-03 | 1.038 |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.989 | 1.853 | -0.136 | -6.84% | 1.806 | 1.428 | 4.456e-03 | 6.042e-03 | 1.356 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 1.788 | 1.920 | 0.132 | 7.38% | 1.547 | 1.442 | 7.538e-03 | 7.050e-03 | 0.935 |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.096 | 5.229 | 0.133 | 2.61% | 4.917 | 4.830 | 1.968e-03 | 1.572e-03 | 0.799 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.545 | 1.817 | 0.272 | 17.61% | 1.345 | 1.430 | 5.829e-03 | 5.737e-03 | 0.984 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 2.121 | 1.818 | -0.303 | -14.29% | 1.923 | 1.384 | 5.676e-03 | 6.073e-03 | 1.070 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 1.771 | 1.871 | 0.100 | 5.65% | 1.470 | 1.458 | 7.507e-03 | 7.141e-03 | 0.951 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.247 | 5.613 | 0.366 | 6.98% | 5.095 | 5.211 | 9.033e-03 | 9.949e-03 | 1.101 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 1.605 | 1.916 | 0.311 | 19.38% | 1.437 | 1.418 | 5.829e-03 | 5.768e-03 | 0.990 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.599 | 1.839 | 0.240 | 15.01% | 1.412 | 1.391 | 6.317e-03 | 6.134e-03 | 0.971 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 1.616 | 1.911 | 0.295 | 18.25% | 1.460 | 1.394 | 7.568e-03 | 7.080e-03 | 0.936 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.366 | 5.586 | 0.220 | 4.10% | 5.123 | 5.142 | 1.031e-02 | 1.233e-02 | 1.196 |
