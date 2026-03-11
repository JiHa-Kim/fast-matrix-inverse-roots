# Benchmark: `dwh4_cubic` Basis Comparison on Light Suite

Date: 2026-03-10

Command:

```
uv run -m polar.main --device cuda --mode suite --suite_shapes light --suite_cases 20 --kappa_G 1e7 --schedule dwh_tuned_fp32 --compare_schedules dwh_tuned_fp32 dwh4_cubic_cheb dwh4_cubic --input_dtype float32 --iter_dtype float32 --tf32_rational_runner --tf32 --exact_verify_device cpu
```

Compared schedules:

- `dwh_tuned_fp32`: 5 tuned rational steps
- `dwh4_cubic_cheb`: 4 tuned rational steps + 1 Gram-cubic direct sigma-map minimax tail, Chebyshev basis with Clenshaw
- `dwh4_cubic`: same optimized tail polynomial in monomial basis

The benchmark uses paired cases: each schedule runs on the same 20 generated matrices per shape.

## Mean Timed Loop (`ms_total_timed`)

| Shape | `dwh_tuned_fp32` | `dwh4_cubic_cheb` | `dwh4_cubic` |
| --- | ---: | ---: | ---: |
| 2048x256 | 11.074 | 11.769 | 8.943 |
| 4096x256 | 13.434 | 13.311 | 12.828 |
| 8192x256 | 25.127 | 23.395 | 22.406 |
| 8192x1024 | 102.002 | 98.423 | 86.382 |
| 16384x1024 | 228.916 | 190.985 | 189.860 |
| 8192x2048 | 272.684 | 250.780 | 245.057 |
| 16384x2048 | 576.172 | 509.924 | 504.770 |

## Worst Exact `kappa(O)`

| Shape | `dwh_tuned_fp32` | `dwh4_cubic_cheb` | `dwh4_cubic` |
| --- | ---: | ---: | ---: |
| 2048x256 | 1.0002089 | 1.0009715 | 1.0009720 |
| 4096x256 | 1.0001510 | 1.0009945 | 1.0019131 |
| 8192x256 | 1.0001042 | 1.0009803 | 1.0019349 |
| 8192x1024 | 1.0002078 | 1.0009774 | 1.0019137 |
| 16384x1024 | 1.0001479 | 1.0009962 | 1.0021828 |
| 8192x2048 | 1.0002933 | 1.0007849 | 1.0007893 |
| 16384x2048 | 1.0002281 | 1.0010010 | 1.0025719 |

## Success Counts

All three schedules were `20/20` successful on every light-suite shape against the robust target `kappa(O) <= 1.0078431`.

## Conclusion

- The direct sigma-map Remez tail works robustly on the 20-case light suite.
- For the same optimized cubic tail, monomial evaluation was consistently faster than Chebyshev/Clenshaw on this workload.
- The quality loss from monomial evaluation remained small and comfortably inside the robust target on the full bank.
- Based on this benchmark, `dwh4_cubic` should point to the optimized monomial-basis tail.
