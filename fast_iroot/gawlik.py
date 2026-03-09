#!/usr/bin/env python3
import dataclasses
import torch
from typing import Tuple

from .ops import symmetrize, chol_with_jitter_fp64

Tensor = torch.Tensor


@torch.no_grad()
def mu_from_alpha(alpha: float, p: int) -> float:
    alpha = float(min(max(alpha, 1e-300), 1.0))
    if alpha >= 1.0 - 1e-15:
        return 1.0
    if p == 2:
        return float(alpha**0.5)
    if p == 4:
        # mu(alpha)^4 = alpha (1 + alpha + alpha^2) / 3
        mu4 = alpha * (1.0 + alpha + alpha * alpha) / 3.0
        mu4 = min(max(mu4, 1e-300), 1.0)
        return float(mu4**0.25)
    
    # Generic Gawlik: mu^p = (alpha - alpha^p) / ((p-1)(1 - alpha))
    mup = (alpha - alpha**p) / ((p - 1) * (1.0 - alpha))
    mup = min(max(mup, 1e-300), 1.0)
    return float(mup**(1.0 / p))


@torch.no_grad()
def alpha_next(alpha: float, mu: float, p: int) -> float:
    # alpha_next = h(1, alpha) = p * mu^{p-1} / (1 + (p-1) * mu^p)
    alpha = float(alpha)
    mu = float(mu)
    num = float(p) * (mu**(p - 1))
    den = 1.0 + float(p - 1) * (mu**p)
    out = num / max(den, 1e-300)
    return float(min(max(out, 1e-300), 1.0))


@torch.no_grad()
def build_w_from_M(
    M: Tensor, alpha: float, p: int, solve_jitter_rel: float
) -> Tuple[Tensor, float, float, float]:
    """
    For the explicit type-(1,0) p-th root minimax step,
      h(z, alpha) = p * mu^{p-1} / (z + (p-1) * mu^p).
    Returns W = h(M, alpha), the shift b = (p-1) * mu^p, mu, and Cholesky jitter shift.
    """
    mu = mu_from_alpha(alpha, p)
    a1 = float(p) * (mu**(p - 1))
    b1 = float(p - 1) * (mu**p)

    if alpha >= 1.0 - 1e-15:
        # If alpha is 1, mu is 1, a1 is p, b1 is p-1.
        # h(z, 1) = p / (z + p - 1).
        # But we also know M should be I, so h(I, 1) = I.
        # Let's check if M is actually I-like.
        # For now, if alpha is very high, just return identity to save Cholesky.
        return torch.eye(M.shape[0], device=M.device, dtype=M.dtype), float(b1), float(mu), 0.0

    # A = M + b1 * I. Since M is symmetric, A is symmetric.
    A = M.clone()
    A.diagonal().add_(b1)
    
    L, shift = chol_with_jitter_fp64(A, jitter_rel=solve_jitter_rel)
    # W = a1 * inv(A) = a1 * cholesky_inverse(L)
    W = a1 * torch.cholesky_inverse(L)
    W = symmetrize(W)
    return W, float(b1), float(mu), float(shift)


@torch.no_grad()
def update_M(M: Tensor, W: Tensor, p: int) -> Tensor:
    # M_{k+1} = h(M_k, alpha_k)^p M_k = W^p M_k
    # If W is identity, M remains M.
    # We check if W is actually the identity from build_w.
    # In build_w, we return a newly created identity if alpha is close to 1.
    # We can use this to skip.
    if alpha_is_identity_heuristic(W):
        return M

    # For large n, MMM is O(n^3). On consumer GPUs, f64 is very slow.
    W32 = W.to(torch.float32)
    M32 = M.to(torch.float32)

    if p == 4:
        W2 = symmetrize(W32 @ W32)
        W4 = symmetrize(W2 @ W2)
        res32 = symmetrize(W4 @ M32)
        return res32.to(torch.float64)
    
    Wk = W32
    for _ in range(p - 1):
        Wk = symmetrize(Wk @ W32)
    res32 = symmetrize(Wk @ M32)
    return res32.to(torch.float64)


def alpha_is_identity_heuristic(W: Tensor) -> bool:
    # Heuristic: check a few diagonal elements and one off-diagonal.
    # If build_w returned torch.eye, this will be exactly 1.0 and 0.0.
    if W.shape[0] == 0:
        return True
    if abs(float(W[0, 0].item()) - 1.0) > 1e-15:
        return False
    if W.shape[0] > 1 and abs(float(W[0, 1].item())) > 1e-15:
        return False
    return True


@dataclasses.dataclass
class ActionCert:
    action_rel_cert: float
    action_rel_exact: float
    resid_M_cert: float
    resid_M_exact: float


@torch.no_grad()
def cert_action_rel_from_M(
    M: Tensor,
    p: int,
    cert_mode: str,
    exact_threshold: int,
) -> ActionCert:
    # We expect M to be in float64 already, but symmetrize it to be sure.
    M = symmetrize(M)
    n = M.shape[0]

    use_exact = (cert_mode == "exact") or (cert_mode == "auto" and n <= exact_threshold)
    inv_p = 1.0 / float(p)

    if use_exact:
        evals = torch.linalg.eigvalsh(M)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        action_rel = max(1.0 - lam_min**inv_p, lam_max**inv_p - 1.0)
        resid_M = max(abs(1.0 - lam_min), abs(lam_max - 1.0))
        return ActionCert(
            action_rel_cert=float(action_rel),
            action_rel_exact=float(action_rel),
            resid_M_cert=float(resid_M),
            resid_M_exact=float(resid_M),
        )

    # ||M - I||_F^2 = ||M||_F^2 - 2*trace(M) + n
    # This is much faster than E = M - I.
    m_fro_sq = torch.linalg.matrix_norm(M, ord="fro").item() ** 2
    m_trace = torch.trace(M).item()
    eta_sq = max(0.0, m_fro_sq - 2.0 * m_trace + n)
    eta = float(eta_sq**0.5)

    if eta >= 1.0:
        action_rel_ub = float("inf")
    else:
        # |(1+eta)^{1/p} - 1| or |1 - (1-eta)^{1/p}|
        action_rel_ub = max(1.0 - (max(0.0, 1.0 - eta))**inv_p, (1.0 + eta)**inv_p - 1.0)
    return ActionCert(
        action_rel_cert=float(action_rel_ub),
        action_rel_exact=float("nan"),
        resid_M_cert=float(eta),
        resid_M_exact=float("nan"),
    )
