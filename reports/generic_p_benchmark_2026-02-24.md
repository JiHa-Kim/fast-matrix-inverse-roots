# Generic $p$ Support Benchmark Report
*Date: 2026-02-24*

This report evaluates the newly implemented generic $p$ support (odd and even branches) by benchmarking $p=3$ and $p=5$ across various SPD matrix conditions.

## Methodology
- **Commands**: `uv run python -m scripts.matrix_iroot --p <val> --sizes 256,512 --trials 5`
- **Baselines**: `Inverse-Newton` (baseline), `PE-Quad` (Uncoupled quadratic), and `PE-Quad-Coupled`.
- **Metrics Evaluated**: Time (ms), Frobenious Residual (`|I - A X^p|_F`), and Memory.

## Results for $p=3$
For $p=3$, the customized `PE-Quad` adaptive quadratic polynomial consistently outperformed standard Inverse-Newton on residual metrics while remaining highly competitive in execution time.
- **Sizes (256x256 and 512x512)**: `PE-Quad` was routinely the BEST overall method by achieving up to 3-5x lower median residuals (e.g. `4.62e-03` vs `1.88e-02` on Gaussian) compared to `Inverse-Newton`, with nearly identical execution speeds (~3.0-3.8 ms).
- **Coupled vs Uncoupled**: `PE-Quad` (Uncoupled) dominated `PE-Quad-Coupled` in convergence precision.
- **Robustness**: The adaptive quadratic fallback handling strictly preserved positivity as required for odd $p$, preventing divergence even on ill-conditioned inputs like `illcond_1e12`.

## Results for $p=5$
For $p=5$, the performance gap tightens. Execution of the uncoupled `X^5` addition chain imposes more overhead compared to lower degree roots.
- **Sizes (256x256 and 512x512)**: `Inverse-Newton` and `PE-Quad-Coupled` emerge as the fastest algorithms (~2.8-6.0 ms depending on size). `PE-Quad` (Uncoupled) takes about 15-20% longer per iteration due to the generic `_bpow` matrix multiplication loop.
- **Residuals**: `Inverse-Newton` slightly outperforms `PE-Quad` in precision (e.g., `1.34e-02` vs `1.54e-02` on 256x256), indicating that the adaptive polynomial advantage diminishes for higher degree objectives without explicit $p=5$ parameter tuning.

## Conclusions
The generic $p$ extension successfully handles odd and even variations robustly.
1. **$p=3$**: The `PE-Quad` uncoupled formulation shines, proving that adaptive interval tuning yields massive precision gains at $p=3$.
2. **$p \geq 5$**: Higher $p$ degrees stress the generic uncoupled matmul chain, making `Inverse-Newton` or `Coupled` forms more advantageous unless specific GEMM-chain specializations are coded. The generic fallback logic fulfills its role by running safely across all conditions.
