# Solver Benchmark A/B Report

Generated: 2026-02-27T18:16:55

A: pe_quad
B: chebyshev

| kind | p | n | k | case | pe_quad_method | chebyshev_method | pe_quad_total_ms | chebyshev_total_ms | delta_ms(B-A) | delta_pct | pe_quad_iter_ms | chebyshev_iter_ms | pe_quad_relerr | chebyshev_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 1 | 1024 | 1 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.108 | 1.822 | -0.286 | -13.57% | 0.794 | 0.502 | 6.714e-03 | 8.179e-03 | 1.218 |
| spd | 1 | 1024 | 1 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 1.909 | 1.867 | -0.042 | -2.20% | 0.801 | 0.502 | 7.080e-03 | 8.057e-03 | 1.138 |
| spd | 1 | 1024 | 16 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.688 | 1.878 | -0.810 | -30.13% | 0.811 | 0.568 | 6.531e-03 | 8.301e-03 | 1.271 |
| spd | 1 | 1024 | 16 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 2.227 | 2.110 | -0.117 | -5.25% | 0.959 | 0.570 | 7.080e-03 | 8.606e-03 | 1.216 |
| spd | 1 | 1024 | 64 | gaussian_spd | PE-Quad-Coupled-Apply | Chebyshev-Apply | 5.786 | 2.107 | -3.679 | -63.58% | 3.421 | 0.608 | 6.592e-03 | 8.118e-03 | 1.231 |
| spd | 1 | 1024 | 64 | illcond_1e6 | PE-Quad-Coupled-Apply | Chebyshev-Apply | 1.983 | 2.271 | 0.288 | 14.52% | 0.826 | 0.809 | 1.044e-02 | 7.751e-03 | 0.742 |
