# Spectral Convergence Results (Readable Summary)

Generated from run artifact: `benchmark_results/runs/2026_02_27/234109_spectral_convergence_step15/spectral_convergence.md`
Generated at: `2026-02-27T23:42:18`

## Run Flags

- `n`: `1024`
- `p`: `2`
- `trials`: `10`
- `l_target`: `0.05`
- `dtype`: `fp64`
- `device`: `auto`
- `seed`: `1234`
- `coeff_mode`: `precomputed`
- `coeff_seed`: `0`
- `coeff_safety`: `1.0`
- `coeff_no_final_safety`: `False`

## Step-Wise Worst-Case Comparison

| step | PE rho(I-Y) | NS rho(I-Y) | better rho | PE cluster90 | NS cluster90 | better cluster90 |
|---:|---:|---:|---|---:|---:|---|
| 0 | 9.50e-01 | 9.50e-01 | Tie | 10.5% | 10.5% | Tie |
| 1 | 3.75e-01 | 8.91e-01 | **PE-Quad** | 19.0% | 36.4% | **Newton-Schulz** |
| 2 | 1.57e-02 | 7.73e-01 | **PE-Quad** | 100.0% | 65.0% | **PE-Quad** |
| 3 | 1.06e-06 | 5.63e-01 | **PE-Quad** | 100.0% | 84.9% | **PE-Quad** |
| 4 | 5.96e-08 | 2.82e-01 | **PE-Quad** | 100.0% | 95.6% | **PE-Quad** |

## Final-Step Outcome

- Final worst-case rho ratio (`NS / PE`): `4.738e+06x` in favor of **PE-Quad**.
- PE final rho: `5.96e-08`; NS final rho: `2.82e-01`.

## Raw Artifacts

- `benchmark_results/runs/2026_02_27/234109_spectral_convergence_step15/spectral_convergence.md`
- `benchmark_results/runs/2026_02_27/234109_spectral_convergence_step15/spectral_convergence.json`
- `benchmark_results/runs/2026_02_27/234109_spectral_convergence_step15/spectral_manifest.json`