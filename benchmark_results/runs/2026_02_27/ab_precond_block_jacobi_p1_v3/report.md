# Solver Benchmark A/B Report

Generated: 2026-02-27T16:01:42

A: jacobi
B: block_jacobi

| kind | p | n | k | case | method | jacobi_total_ms | block_jacobi_total_ms | delta_ms(B-A) | delta_pct | jacobi_iter_ms | block_jacobi_iter_ms | jacobi_relerr | block_jacobi_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonspd | 1 | 256 | 256 | gaussian_shifted | PE-Quad-Coupled-Apply | 2.257 | 2.150 | -0.107 | -4.74% | 2.085 | 1.058 | 3.362e-05 | 7.111e-03 | 211.511 |
| nonspd | 1 | 256 | 256 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.551 | 2.057 | 0.506 | 32.62% | 1.424 | 0.976 | 5.219e-03 | 4.852e-03 | 0.930 |
| nonspd | 1 | 256 | 256 | similarity_posspec | PE-Quad-Coupled-Apply | 1.498 | 2.337 | 0.839 | 56.01% | 1.370 | 1.323 | 2.801e-05 | 3.576e-05 | 1.277 |
| nonspd | 1 | 256 | 256 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 1.496 | 2.325 | 0.829 | 55.41% | 1.368 | 1.322 | 3.777e-04 | 8.507e-04 | 2.252 |
| nonspd | 1 | 512 | 512 | gaussian_shifted | PE-Quad-Coupled-Apply | 4.902 | 5.979 | 1.077 | 21.97% | 4.765 | 4.773 | 5.188e-04 | 5.188e-04 | 1.000 |
| nonspd | 1 | 512 | 512 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.129 | 2.299 | 1.170 | 103.63% | 0.993 | 1.095 | 5.219e-03 | 4.456e-03 | 0.854 |
| nonspd | 1 | 512 | 512 | similarity_posspec | PE-Quad-Coupled-Apply | 3.299 | 4.317 | 1.018 | 30.86% | 3.160 | 3.147 | 5.722e-04 | 5.684e-04 | 0.993 |
| nonspd | 1 | 512 | 512 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 3.328 | 4.184 | 0.856 | 25.72% | 3.150 | 3.199 | 6.989e-03 | 1.587e-02 | 2.271 |
| nonspd | 1 | 1024 | 1 | gaussian_shifted | PE-Quad-Coupled-Apply | 6.208 | 7.682 | 1.474 | 23.74% | 6.034 | 6.321 | 0.000e+00 | 0.000e+00 | nan |
| nonspd | 1 | 1024 | 1 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.481 | 2.637 | 1.156 | 78.06% | 1.333 | 1.352 | 5.219e-03 | 6.042e-03 | 1.158 |
| nonspd | 1 | 1024 | 1 | similarity_posspec | PE-Quad-Coupled-Apply | 4.908 | 5.845 | 0.937 | 19.09% | 4.757 | 4.746 | 0.000e+00 | 0.000e+00 | nan |
| nonspd | 1 | 1024 | 1 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.013 | 5.860 | 0.847 | 16.90% | 4.860 | 4.771 | 2.365e-03 | 2.045e-03 | 0.865 |
| nonspd | 1 | 1024 | 16 | gaussian_shifted | PE-Quad-Coupled-Apply | 6.061 | 7.059 | 0.998 | 16.47% | 5.909 | 5.846 | 5.302e-04 | 5.150e-04 | 0.971 |
| nonspd | 1 | 1024 | 16 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.507 | 2.450 | 0.943 | 62.57% | 1.353 | 1.374 | 5.310e-03 | 6.104e-03 | 1.150 |
| nonspd | 1 | 1024 | 16 | similarity_posspec | PE-Quad-Coupled-Apply | 5.223 | 6.366 | 1.143 | 21.88% | 5.076 | 5.080 | 5.493e-04 | 5.722e-04 | 1.042 |
| nonspd | 1 | 1024 | 16 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.286 | 6.118 | 0.832 | 15.74% | 5.061 | 5.080 | 1.129e-02 | 2.686e-02 | 2.379 |
| nonspd | 1 | 1024 | 64 | gaussian_shifted | PE-Quad-Coupled-Apply | 6.047 | 7.158 | 1.111 | 18.37% | 5.900 | 5.990 | 5.684e-04 | 5.684e-04 | 1.000 |
| nonspd | 1 | 1024 | 64 | nonnormal_upper | PE-Quad-Coupled-Apply | 1.518 | 2.572 | 1.054 | 69.43% | 1.368 | 1.418 | 5.310e-03 | 6.134e-03 | 1.155 |
| nonspd | 1 | 1024 | 64 | similarity_posspec | PE-Quad-Coupled-Apply | 5.187 | 6.314 | 1.127 | 21.73% | 5.037 | 5.145 | 6.409e-04 | 6.371e-04 | 0.994 |
| nonspd | 1 | 1024 | 64 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 5.233 | 6.227 | 0.994 | 18.99% | 5.083 | 5.046 | 1.855e-02 | 3.735e-02 | 2.013 |
| nonspd | 1 | 1024 | 1024 | gaussian_shifted | PE-Quad-Coupled-Apply | 8.411 | 10.684 | 2.273 | 27.02% | 8.234 | 8.242 | 5.875e-04 | 5.913e-04 | 1.006 |
| nonspd | 1 | 1024 | 1024 | nonnormal_upper | PE-Quad-Coupled-Apply | 2.394 | 3.362 | 0.968 | 40.43% | 2.246 | 2.251 | 5.676e-03 | 6.439e-03 | 1.134 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec | PE-Quad-Coupled-Apply | 6.811 | 7.825 | 1.014 | 14.89% | 6.666 | 6.628 | 6.638e-04 | 6.599e-04 | 0.994 |
| nonspd | 1 | 1024 | 1024 | similarity_posspec_hard | PE-Quad-Coupled-Apply | 6.851 | 7.707 | 0.856 | 12.49% | 6.693 | 6.590 | 1.373e-02 | 3.467e-02 | 2.525 |
| spd | 1 | 256 | 256 | gaussian_spd | PE-Quad-Coupled-Apply | 2.164 | 2.764 | 0.600 | 27.73% | 0.551 | 0.578 | 4.730e-03 | 4.822e-03 | 1.019 |
| spd | 1 | 256 | 256 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.755 | 2.810 | 1.055 | 60.11% | 0.649 | 0.669 | 5.798e-03 | 4.944e-03 | 0.853 |
| spd | 1 | 512 | 512 | gaussian_spd | PE-Quad-Coupled-Apply | 1.830 | 3.450 | 1.620 | 88.52% | 0.572 | 0.560 | 5.707e-03 | 4.028e-03 | 0.706 |
| spd | 1 | 512 | 512 | illcond_1e6 | PE-Quad-Coupled-Apply | 1.823 | 2.737 | 0.914 | 50.14% | 0.683 | 0.588 | 4.608e-03 | 6.653e-03 | 1.444 |
| spd | 1 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | 1.990 | 3.291 | 1.301 | 65.38% | 0.814 | 0.794 | 6.714e-03 | 1.013e-02 | 1.509 |
| spd | 1 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.056 | 3.238 | 1.182 | 57.49% | 0.854 | 0.807 | 7.080e-03 | 7.233e-03 | 1.022 |
| spd | 1 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | 1.919 | 3.419 | 1.500 | 78.17% | 0.808 | 0.812 | 6.531e-03 | 1.031e-02 | 1.579 |
| spd | 1 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.067 | 3.987 | 1.920 | 92.89% | 0.817 | 0.807 | 7.080e-03 | 7.050e-03 | 0.996 |
| spd | 1 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | 2.720 | 3.361 | 0.641 | 23.57% | 0.837 | 0.809 | 6.592e-03 | 1.025e-02 | 1.555 |
| spd | 1 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.063 | 3.409 | 1.346 | 65.24% | 0.812 | 0.809 | 1.044e-02 | 7.080e-03 | 0.678 |
| spd | 1 | 1024 | 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.472 | 4.888 | 1.416 | 40.78% | 2.139 | 2.123 | 4.150e-03 | 7.629e-03 | 1.838 |
| spd | 1 | 1024 | 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 3.414 | 4.227 | 0.813 | 23.81% | 2.042 | 2.044 | 5.371e-03 | 5.341e-03 | 0.994 |
