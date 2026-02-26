# 2026-02-26 Benchmark Artifacts

## Folders

- `idea4_precond_t20/`
  - Solve preconditioner ablation logs (`p in {1,2,4}`, `k in {1,16,64}`, 20 trials).
  - Aggregated summary: `summary_coupled_apply.md`.

- `idea4_precond_iroot_t20/`
  - IRoot preconditioner ablation logs (`p in {1,2,4}`, 20 trials).
  - Aggregated summary: `summary_pe_quad_coupled.md`.

- `idea4_gram_precond_t20/`
  - Gram-path parity/timing check for `precond_gram_spd`.
  - Summary: `summary.md`.
