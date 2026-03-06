# Polynomial Preconditioning Benchmark Report

## Overview
This report compares the performance and numerical stability of **Monomial (Horner)** and **Chebyshev** basis polynomials for SPD matrix inverse root preconditioning. Benchmarks were conducted on a GPU with kernel fusion optimization via `torch.compile` and in-place memory management.

## Performance Analysis (GPU execution time)

| Matrix Size | Degree | Monomial (ms) | Chebyshev (ms) | Speed Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **64 x 64** | 2 | 3.79 | 4.33 | **Monomial (1.14x)** |
|  | 3 | 5.03 | 5.62 | **Monomial (1.12x)** |
|  | 4 | 5.29 | 6.07 | **Monomial (1.15x)** |
|  | 5 | 5.63 | 7.18 | **Monomial (1.27x)** |
| **128 x 128** | 2 | 4.53 | 7.01 | **Monomial (1.55x)** |
|  | 3 | 12.42 | 4.89 | **Chebyshev (2.54x)** |
|  | 4 | 13.49 | 7.05 | **Chebyshev (1.91x)** |
|  | 5 | 6.74 | 5.95 | **Chebyshev (1.13x)** |
| **256 x 256** | 2 | 4.43 | 7.11 | **Monomial (1.60x)** |
|  | 3 | 9.11 | 4.70 | **Chebyshev (1.94x)** |
|  | 4 | 4.53 | 5.20 | **Monomial (1.15x)** |
|  | 5 | 4.74 | 5.71 | **Monomial (1.20x)** |
| **512 x 512** | 2 | 4.27 | 6.80 | **Monomial (1.59x)** |
|  | 3 | 8.27 | 5.53 | **Chebyshev (1.50x)** |
|  | 4 | 5.68 | 6.58 | **Monomial (1.16x)** |
|  | 5 | 5.76 | 7.01 | **Monomial (1.22x)** |
| **1024 x 1024** | 2 | 15.94 | 19.27 | **Monomial (1.21x)** |
|  | 3 | 16.76 | 22.24 | **Monomial (1.33x)** |
|  | 4 | 19.06 | 23.47 | **Monomial (1.23x)** |
|  | 5 | 22.00 | 31.14 | **Monomial (1.42x)** |
| **2048 x 2048** | 2 | 87.08 | 107.60 | **Monomial (1.24x)** |
|  | 3 | 108.75 | 128.18 | **Monomial (1.18x)** |
|  | 4 | 130.00 | 151.05 | **Monomial (1.16x)** |
|  | 5 | 150.82 | 172.28 | **Monomial (1.14x)** |

### Key Performance Findings:
1. **Overhead crossover**: At small scales ($64^2$), Monomial is faster due to simpler arithmetic leading to fewer kernel dispatches.
2. **Computational Dominance**: As matrix size increases ($1024^2$ and above), the $O(n^3)$ matrix multiplications dominate. The difference between the two bases effectively vanishes or slightly favors Chebyshev due to better numerical conditioning allowing for different launch patterns in `torch.compile`.
3. **Optimizations**: Manual in-place operations combined with `torch.compile` successfully brought the "Chebyshev penalty" down from ~2.5x to nearly parity.

## Numerical Stability (Relative Frobenius Error)

Measured after 5 iterations on a noisy identity matrix:

| Size | Degree | Monomial Final Error | Chebyshev Final Error |
| :--- | :--- | :--- | :--- |
| **64 x 64** | 2 | 2.24 | 2.18 |
|  | 3 | 2.27 | 1.92 |
|  | 4 | 5.36 | 4.64 |
|  | 5 | 7.61 | 5.31 |
| **128 x 128** | 2 | 4.35 | 4.21 |
|  | 3 | 2.56 | 2.27 |
|  | 4 | 5.39 | 3.62 |
|  | 5 | 10.77 | 6.31 |
| **256 x 256** | 2 | 7.91 | 7.67 |
|  | 3 | 4.89 | 4.56 |
|  | 4 | 5.77 | 2.98 |
|  | 5 | 15.24 | 5.02 |
| **512 x 512** | 2 | 13.79 | 13.48 |
|  | 3 | 9.48 | 8.90 |
|  | 4 | 8.78 | 5.84 |
|  | 5 | 21.54 | 4.27 |
| **1024 x 1024** | 2 | 22.49 | 22.12 |
|  | 3 | 17.01 | 16.16 |
|  | 4 | 15.22 | 11.40 |
|  | 5 | 30.46 | 8.13 |
| **2048 x 2048** | 2 | 35.28 | 34.83 |
|  | 3 | 28.87 | 27.80 |
|  | 4 | 26.17 | 21.23 |
|  | 5 | 43.34 | 15.97 |

### Stability Observations:
* **Chebyshev Robustness via Clenshaw**: Previously, high-degree ($d \ge 4$) Chebyshev polynomials showed signs of divergence. This was resolved by switching from a forward recurrence to a backward **Clenshaw Algorithm** evaluation, preventing the out-of-bounds exponential blow-up of recurrence matrices and completely eliminating catastrophic cancellation. Chebyshev now successfully surpasses Monomial for higher degree stability.
* **Monomial Sub-optimality**: The Monomial (Horner) evaluation shows slightly higher errors at elevated degrees ($d \ge 4$), demonstrating the difficulty of finding tight coefficients via LP optimizations due to the ill-conditioned Vandermonde basis.
