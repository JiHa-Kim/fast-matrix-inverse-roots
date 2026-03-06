import torch


def fro_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a, ord="fro")


def jacobi_init(B: torch.Tensor, jacobi_eps: float) -> torch.Tensor:
    d = torch.diagonal(B).to(torch.float32)
    inv_sqrt = torch.rsqrt(d + jacobi_eps)
    Z = torch.diag(inv_sqrt).to(torch.bfloat16)
    return Z


def choose_beta(S: torch.Tensor, mode: str = "fro") -> torch.Tensor:
    if mode == "fro":
        return fro_norm(S).clamp_min(1.0)
    if mode == "trace":
        n = S.shape[0]
        return (torch.trace(S).abs() / n).clamp_min(1.0)
    if mode == "maxdiag":
        return torch.max(torch.diagonal(S)).clamp_min(1.0)
    raise ValueError("beta mode must be fro|trace|maxdiag")


def apply_poly_right_mono(
    Z: torch.Tensor, S: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """
    Compute Z q(S) with monomial Horner:
      Y = a[d] Z
      for k=d-1..0: Y = Y S + a[k] Z
    """
    d = a.numel() - 1
    Y = a[d] * Z
    for k in range(d - 1, -1, -1):
        Y = Y @ S
        Y = Y + a[k] * Z
    return Y


def apply_poly_right_cheb(
    Z: torch.Tensor, S: torch.Tensor, c: torch.Tensor, ell: float
) -> torch.Tensor:
    """
    Evaluate q(S) = sum_{k=0}^d c[k] T_k(t(S)) on [ell,1], but apply on the right to Z.
    We avoid forming t explicitly by using:
      t = alpha S + beta I,  alpha = 2/(1-ell), beta = -(1+ell)/(1-ell)

    We use Clenshaw's algorithm (backward recurrence) for vastly improved numerical stability
    over forward recurrence, particularly when S has eigenvalues outside the safe [ell, 1] range:
      B_k = c_k Z + 2 B_{k+1} t - B_{k+2}   (for k = d down to 1)
      out = c_0 Z + B_1 t - B_2
    """
    d = c.numel() - 1
    if d == 0:
        return c[0] * Z

    alpha = 2.0 / (1.0 - ell)
    beta = -(1.0 + ell) / (1.0 - ell)

    B_k2 = torch.zeros_like(Z)
    B_k1 = c[d] * Z  # Start at k=d

    for k in range(d - 1, 0, -1):
        B_k1_S = B_k1 @ S
        B_k = c[k] * Z + 2.0 * (alpha * B_k1_S + beta * B_k1) - B_k2
        B_k2 = B_k1
        B_k1 = B_k

    # Final step for k=0 (the coefficient of B_1 is 1x, not 2x for standard Chebyshev)
    B_k1_S = B_k1 @ S
    out = c[0] * Z + (alpha * B_k1_S + beta * B_k1) - B_k2

    return out
