# Solver Benchmark A/B Report

Generated: 2026-02-27T18:19:05

A: cheb_klt24
B: cheb_klt16

| kind | p | n | k | case | method | cheb_klt24_total_ms | cheb_klt16_total_ms | delta_ms(B-A) | delta_pct | cheb_klt24_iter_ms | cheb_klt16_iter_ms | cheb_klt24_relerr | cheb_klt16_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1 | 1024 | 1 | gaussian_spd | Chebyshev-Apply | 1.830 | 1.893 | 0.063 | 3.44% | 0.504 | 0.344 | 8.179e-03 | 8.301e-03 | 1.015 |
| spd | 1 | 1024 | 1 | illcond_1e6 | Chebyshev-Apply | 1.594 | 1.473 | -0.121 | -7.59% | 0.507 | 0.344 | 8.057e-03 | 8.362e-03 | 1.038 |
| spd | 1 | 1024 | 16 | gaussian_spd | Chebyshev-Apply | 1.877 | 1.732 | -0.145 | -7.73% | 0.567 | 0.392 | 8.301e-03 | 8.301e-03 | 1.000 |
| spd | 1 | 1024 | 16 | illcond_1e6 | Chebyshev-Apply | 1.843 | 1.670 | -0.173 | -9.39% | 0.566 | 0.389 | 8.606e-03 | 8.667e-03 | 1.007 |
| spd | 1 | 1024 | 64 | gaussian_spd | Chebyshev-Apply | 1.933 | 4.423 | 2.490 | 128.82% | 0.611 | 2.506 | 8.118e-03 | 8.179e-03 | 1.008 |
| spd | 1 | 1024 | 64 | illcond_1e6 | Chebyshev-Apply | 4.874 | 1.798 | -3.076 | -63.11% | 3.629 | 0.393 | 7.751e-03 | 7.751e-03 | 1.000 |
