# Chebyshev Direct Apply (`Z = A^{-1/p} B`)

The `apply_inverse_proot_chebyshev` method approximates $Z = f(A)B$ where $f(x)=x^{-1/p}$ without explicitly materializing the dense matrix $A^{-1/p}$.

## Why It Exists

For $B \in \mathbb{R}^{n \times k}$ with $k \ll n$:

- **Standard Route**: Forming $A^{-1/p}$ takes $O(n^3)$ operations.
- **Direct Apply Route**: Using polynomial evaluation takes $O(d \cdot n^2 k)$ operations, where $d$ is the polynomial degree.

## Approximation Model

- **Interval**: Approximates $x^{-1/p}$ on the spectral interval $[\lambda_{min}, \lambda_{max}]$.
- **Coefficients**: Computed using discrete Chebyshev orthogonality and cached via `lru_cache`.
- **Evaluation**: Uses the **Clenshaw recurrence** (a numerically stable algorithm for evaluating polynomial combinations) for matrix evaluation.

## Clenshaw Recurrence

With an affine map $t(A)$ from $[\lambda_{min}, \lambda_{max}]$ to $[-1, 1]$, the recurrence is evaluated backward:

$$
y_k = c_k B + 2 t(A)y_{k+1} - y_{k+2}
$$

The final output is:

$$
Z = c_0 B + t(A)y_1 - y_2
$$

## Implementation Details

- **Memory Efficiency**: Requires only a few tensors shaped like $B$. No $n \times n$ inverse-root buffer is allocated.
- **Workspace**: Supports `InverseApplyAutoWorkspace` for memory reuse.
- **CUDA Graphs**: Optimized with CUDA Graph replay for minimal launch overhead in repeated applies.

## Matrix-Free Path for Gram Matrices

A specialized path `apply_inverse_proot_chebyshev_gram` exists for wide **Gram matrices** (matrices formed as $A = G^T G$).

- **Complexity**: Instead of $O(n^2 k)$, it operates in $O(b \cdot n k)$ where $b$ is the inner dimension of $G$.
- **Performance**: For $b \ll n$, this is consistently $>10\times$ faster than dense PE-Quad methods and dramatically reduces memory usage by avoiding the formation of $G^T G$.

## Practical Behavior

- **Scaling**: At $n=2048$, Chebyshev is typically the fastest path for small $k$.
- **Precision**: Highly dependent on the condition number and polynomial degree. Default degrees are tuned for common ML ranges.
