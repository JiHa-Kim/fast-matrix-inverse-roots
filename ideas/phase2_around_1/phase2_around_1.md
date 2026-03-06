# Phase 2: Local Minimax Refinement Grid Design

## 1. The Local Minimax Problem
After Phase 1, the eigenvalues of the preconditioned matrix $S$ are tightly clustered around 1.0. Let the residual be $\delta_F = \|S - I\|_F$. We define a symmetric interval $[1-\rho, 1+\rho]$ where $\rho \approx \gamma \delta_F$ (with $\gamma$ acting as a safety inflation factor).

The goal of the Phase 2 local step is to find a polynomial $q(x)$ of degree $d$ that minimizes the maximum relative error for the target function $f(x) = x^{-1/r}$ on this local interval:
$$ \min_{q} \max_{x \in [1-\rho, 1+\rho]} | 1 - x^{1/r} q(x) | $$

## 2. The Optimal Grid (Proxy Set) $X_{\text{proxy}}$
To solve this continuous minimax problem computationally, we discretize the interval $[1-\rho, 1+\rho]$ into a proxy set $X_{\text{proxy}}$. To ensure the polynomial is robust under the severe quantization constraints of `bf16` hardware, the optimal grid is constructed as the union of three distinct point sets:

### a) Logarithmic Grid ($X_{\text{log}}$)
Points geometrically spaced to provide higher density near the lower bound, where errors can have an outsized impact on convergence.
$$ X_{\text{log}} = \left\{ x_i \in [1-\rho, 1+\rho] \mid x_i = (1-\rho) \cdot \left(\frac{1+\rho}{1-\rho}\right)^{\frac{i}{N_{\text{log}}-1}}, \quad i=0, \dots, N_{\text{log}}-1 \right\} $$

### b) Linear Grid ($X_{\text{lin}}$)
Uniformly distributed points to provide a dense, even baseline coverage across the entire interval.
$$ X_{\text{lin}} = \left\{ x_i \in [1-\rho, 1+\rho] \mid x_i = 1-\rho + i \cdot \frac{2\rho}{N_{\text{lin}}-1}, \quad i=0, \dots, N_{\text{lin}}-1 \right\} $$

### c) Exact BF16 Representables ($X_{\text{bf16}}$)
The most critical component for hardware-aware design. We explicitly evaluate every single exact `bf16` 16-bit floating-point value that falls within the interval. This guarantees that the linear programming constraints perfectly align with the exact discrete points the hardware will physically operate on.
$$ X_{\text{bf16}} = \{ x \in \mathbb{F}_{\text{bf16}} \mid 1-\rho \le x \le 1+\rho \} $$

The final proxy set is the unique, sorted union of these sets:
$$ X_{\text{proxy}} = X_{\text{log}} \cup X_{\text{lin}} \cup X_{\text{bf16}} $$

## 3. Linear Programming Formulation
We solve the minimax problem over $X_{\text{proxy}}$ using a linear programming (LP) solver. To ensure numerical stability (especially for higher degrees), we formulate the problem in the Chebyshev basis $T_k(t)$, where $t \in [-1, 1]$ is the affine mapping of $x \in [1-\rho, 1+\rho]$:
$$ t(x) = \frac{2x - (1-\rho + 1+\rho)}{(1+\rho) - (1-\rho)} = \frac{x - 1}{\rho} $$

The polynomial is $q(x) = \sum_{k=0}^{d} c_k T_k(t(x))$.
The objective is to minimize the slack variable $\delta$.

For each point $x_i \in X_{\text{proxy}}$, we evaluate the Chebyshev Vandermonde matrix row $V_i = [T_0(t(x_i)), \dots, T_d(t(x_i))]$ and scale it by $s_i = x_i^{1/r}$ to get $G_i = s_i V_i$.
The maximum residual constraint $|1 - x_i^{1/r} q(x_i)| \le \delta$ splits into two linear inequalities:
$$ x_i^{1/r} q(x_i) - \delta \le 1 $$
$$ -x_i^{1/r} q(x_i) - \delta \le -1 $$

In matrix form, the LP is defined as:
$$ \text{Minimize } \delta $$
$$ \text{Subject to:} $$
$$ G c - \delta \mathbf{1} \le \mathbf{1} $$
$$ -G c - \delta \mathbf{1} \le -\mathbf{1} $$
where $c = [c_0, \dots, c_d]^T$ are the polynomial coefficients we are optimizing.

## 4. BF16-in-the-Loop Refinement
The LP solver identifies the optimal coefficients $c^*$ assuming perfect $fp64$ continuous arithmetic. However, executing $q(x)$ in pure `bf16` introduces quantization noise at every single multiply-add operation during evaluation. 

To bridge this gap and find the absolute hardware-optimal coefficients, we perform a derivative-free pattern search (coordinate descent) initialized at $c^*$. We explicitly simulate Horner or Clenshaw evaluation with native `bf16` rounding (`bf16_round_f32`) applied at every computational step. 

We search for coefficients that minimize the exact hardware certificate error:
$$ \text{err}(c) = \max_{x \in X_{\text{bf16}}} \left| 1 - \text{round}_{\text{bf16}}\left(x \cdot \text{round}_{\text{bf16}}(q(x)^r)\right) \right| $$

This yields the final hardware-optimal Phase 2 polynomial that exhibits maximum robustness against quantization errors.

## 5. The Danger of Overfitting on Sparse bf16 Grids
While explicitly adding exact `bf16` representables to the proxy set is strictly necessary to guarantee zero overshoot in hardware, evaluating **only** on `bf16` representables is a catastrophic failure mode for the linear programming solver. 

In extremely tight intervals, the number of representable `bf16` points becomes pathologically sparse. For example, in the interval $[0.99, 1.01]$, there are exactly **4 representable `bf16` points**. 

If we attempt to optimize a higher-degree polynomial (e.g. $d=5$) using only those 4 points as constraints, the LP solver becomes severely underconstrained. The polynomial will overfit exactly to those 4 hardware points but exhibit massive, chaotic oscillations (Runge's Phenomenon) in the continuous spaces between them. Because the preconditioned matrix $S$ has a continuous spectrum of true real eigenvalues (not limited to 16-bit rounded numbers), the mathematical action of the polynomial operates on the continuous spectrum before subsequent matrix multiplication rounding.

Our empirical tests on $[0.99, 1.01]$ with $d=5$ showed:
- **Continuous Error (Dense Proxy Grid):** $0.00779$
- **Continuous Error (Sparse `bf16` Only):** $1.73 \times 10^{13}$

Thus, a dense combination of logarithmic, linear, and exact `bf16` points ($X_{\text{proxy}}$) is not an arbitrary heuristic—it is a mandatory mathematical constraint for stability.
