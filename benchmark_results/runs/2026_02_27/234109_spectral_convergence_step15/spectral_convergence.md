# Spectral Convergence Benchmark

Generated: 2026-02-27T23:42:18

## Run Configuration

- n: `1024`
- p: `2`
- trials: `10`
- l_target: `0.05`
- dtype: `fp64`
- device: `cuda`
- seed: `1234`
- coeff_mode: `precomputed`
- coeff_seed: `0`
- coeff_safety: `1.0`
- coeff_no_final_safety: `False`
- pe_steps: `4`

## Coefficients (PE-Quad)

| Step | a | b | c |
|---:|---:|---:|---:|
| 0 | 3.902148485 | -7.590706825 | 4.860831261 |
| 1 | 1.937780857 | -1.349293113 | 0.410987377 |
| 2 | 1.875123501 | -1.250201106 | 0.375077546 |
| 3 | 1.874953985 | -1.249907970 | 0.374954015 |

## PE-Quad (Worst Case Over Trials)

| Step | Min λ | Max λ | Mean λ | ρ(I-Y) | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.5% | 1.1% |
| 1 | 0.6247 | 1.3742 | 0.9659 | 3.75e-01 | 19.0% | 2.0% |
| 2 | 0.9843 | 1.0157 | 1.0001 | 1.57e-02 | 100.0% | 99.2% |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.06e-06 | 100.0% | 100.0% |
| 4 | 1.0000 | 1.0000 | 1.0000 | 5.96e-08 | 100.0% | 100.0% |

## Newton-Schulz (Worst Case Over Trials)

| Step | Min λ | Max λ | Mean λ | ρ(I-Y) | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.5% | 1.1% |
| 1 | 0.1088 | 1.0000 | 0.7206 | 8.91e-01 | 36.4% | 12.0% |
| 2 | 0.2273 | 1.0000 | 0.8667 | 7.73e-01 | 65.0% | 38.7% |
| 3 | 0.4369 | 1.0000 | 0.9509 | 5.63e-01 | 84.9% | 66.9% |
| 4 | 0.7176 | 1.0000 | 0.9880 | 2.82e-01 | 95.6% | 85.8% |

## Reproducibility

This report is paired with:
- `spectral_convergence.json` (raw per-step rows)
- `spectral_manifest.json` (run metadata + reproducibility fingerprint)
- `.sha256` sidecars for all output files
