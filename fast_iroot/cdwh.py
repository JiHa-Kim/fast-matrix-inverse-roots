import math
from typing import Tuple

import torch
from torch import Tensor

from fast_iroot.core import RunSummary
from fast_iroot.utils import symmetrize, cuda_time_ms, safe_exp, acosh_exp


@torch.no_grad()
def gram_xtx_chunked_fp64(X: Tensor, chunk_rows: int) -> Tensor:
    m, n = X.shape
    device = X.device
    S = torch.zeros((n, n), device=device, dtype=torch.float64)

    for i in range(0, m, chunk_rows):
        Xi = X[i : i + chunk_rows]
        Xi = Xi.float().to(torch.float64)
        S.addmm_(Xi.T, Xi)

    return symmetrize(S)


@torch.no_grad()
def chol_with_jitter_fp64(
    A: Tensor,
    jitter_rel: float,
    max_tries: int = 8,
) -> Tuple[Tensor, float]:
    A = symmetrize(A.to(torch.float64))
    if not torch.isfinite(A).all():
        raise RuntimeError("non-finite matrix before Cholesky")

    n = A.shape[0]
    Id_mat = torch.eye(n, device=A.device, dtype=torch.float64)

    scale = float((torch.trace(A).abs() / max(n, 1)).item())
    base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)

    delta = 0.0
    for _ in range(max_tries):
        At = A if delta == 0.0 else (A + delta * Id_mat)
        L, info = torch.linalg.cholesky_ex(At)
        if int(info.item()) == 0:
            return L, float(delta)
        delta = base if delta == 0.0 else 2.0 * delta

    raise RuntimeError("Cholesky failed even after jitter escalation")


def dwh_coeffs_from_ell(ell: float) -> Tuple[float, float, float]:
    ell = float(min(max(ell, 1e-300), 1.0))
    ell2 = ell * ell

    d = (4.0 * (1.0 - ell2) / (ell2 * ell2)) ** (1.0 / 3.0)
    a = math.sqrt(1.0 + d) + 0.5 * math.sqrt(
        8.0 - 4.0 * d + 8.0 * (2.0 - ell2) / (ell2 * math.sqrt(1.0 + d))
    )
    b = 0.25 * (a - 1.0) * (a - 1.0)
    c = a + b - 1.0
    return float(a), float(b), float(c)


def dwh_ell_next(ell: float, a: float, b: float, c: float) -> float:
    ell = float(ell)
    return float(ell * (a + b * ell * ell) / (1.0 + c * ell * ell))


@torch.no_grad()
def build_q_from_s(
    S: Tensor,
    a: float,
    b: float,
    c: float,
    jitter_rel: float,
) -> Tuple[Tensor, Tensor, float]:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]
    Id_mat = torch.eye(n, device=S.device, dtype=torch.float64)

    M = symmetrize(Id_mat + float(c) * S)
    R = symmetrize(float(a) * Id_mat + float(b) * S)

    L, shift = chol_with_jitter_fp64(M, jitter_rel=jitter_rel)
    Q = torch.cholesky_solve(R, L)
    Q = symmetrize(Q)
    return Q, L, float(shift)


@torch.no_grad()
def apply_dwh_update_chunked(
    X: Tensor,
    L: Tensor,
    a: float,
    b: float,
    c: float,
    rhs_chunk_rows: int,
    out_dtype: torch.dtype,
) -> Tensor:
    alpha = float(b / c)
    beta = float(a - b / c)

    m, n = X.shape
    device = X.device
    X_next = torch.empty((m, n), device=device, dtype=out_dtype)

    for i in range(0, m, rhs_chunk_rows):
        Xi = X[i : i + rhs_chunk_rows].float().to(torch.float64)  # (b, n)
        Yi_t = torch.cholesky_solve(Xi.T.contiguous(), L)  # (n, b)
        Yi = Yi_t.T  # (b, n)
        Zi = alpha * Xi + beta * Yi
        X_next[i : i + rhs_chunk_rows] = Zi.to(out_dtype)

    return X_next


@torch.no_grad()
def cert_from_s(
    S: Tensor,
    cert_mode: str,
    exact_threshold: int,
    chol_jitter_rel: float,
) -> Tuple[float, float, float]:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    use_exact = (cert_mode == "exact") or (cert_mode == "auto" and n <= exact_threshold)

    if use_exact:
        evals = torch.linalg.eigvalsh(S)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        kappa_O = float(math.sqrt(lam_max / lam_min))
        return float(kappa_O), float(kappa_O), 0.0

    L, shift = chol_with_jitter_fp64(S, jitter_rel=chol_jitter_rel)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()

    a = max(float((torch.trace(S) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    kappa_O_ub = safe_exp(eta_ub)
    return float(kappa_O_ub), float("nan"), float(shift)


@torch.no_grad()
def run_cdwh(
    G_storage: Tensor,
    kappa_G_upper: float,
    target_kappa_O: float,
    max_steps: int,
    iter_dtype: torch.dtype,
    cert_mode: str,
    exact_threshold: int,
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    solve_jitter_rel: float,
    cert_jitter_rel: float,
    tf32: bool,
) -> RunSummary:
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    X = G_storage.to(dtype=iter_dtype)
    ell = 1.0 / float(kappa_G_upper)

    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    ms_cert_sum = 0.0
    guards = 0

    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
    ms_gram_sum += ms_gram

    final_kO_cert = float("inf")
    final_kO_exact = float("nan")
    final_kO_pred = float("inf")

    for it in range(1, max_steps + 1):
        a, b, c = dwh_coeffs_from_ell(ell)

        ms_solve, (Q, L, shift) = cuda_time_ms(
            lambda: build_q_from_s(
                S=S,
                a=a,
                b=b,
                c=c,
                jitter_rel=solve_jitter_rel,
            )
        )
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

        S_next = symmetrize(Q @ S @ Q)
        S = S_next

        ms_upd, X = cuda_time_ms(
            lambda: apply_dwh_update_chunked(
                X=X,
                L=L,
                a=a,
                b=b,
                c=c,
                rhs_chunk_rows=rhs_chunk_rows,
                out_dtype=iter_dtype,
            )
        )
        ms_upd_sum += ms_upd

        ell = dwh_ell_next(ell, a, b, c)
        final_kO_pred = 1.0 / max(ell, 1e-300)

        ms_cert, (kO_cert, kO_exact, cert_shift) = cuda_time_ms(
            lambda: cert_from_s(
                S=S,
                cert_mode=cert_mode,
                exact_threshold=exact_threshold,
                chol_jitter_rel=cert_jitter_rel,
            )
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)

        final_kO_cert = float(kO_cert)
        final_kO_exact = float(kO_exact)

        if final_kO_cert <= target_kappa_O:
            ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
            return RunSummary(
                success=True,
                final_kO_cert=final_kO_cert,
                final_kO_exact=final_kO_exact,
                final_kO_pred=final_kO_pred,
                steps=it,
                guards=guards,
                ms_total=ms_total,
                ms_details={
                    "gram": ms_gram_sum,
                    "solve": ms_solve_sum,
                    "upd": ms_upd_sum,
                    "cert": ms_cert_sum,
                },
            )

    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
    return RunSummary(
        success=False,
        final_kO_cert=final_kO_cert,
        final_kO_exact=final_kO_exact,
        final_kO_pred=final_kO_pred,
        steps=max_steps,
        guards=guards,
        ms_total=ms_total,
        ms_details={
            "gram": ms_gram_sum,
            "solve": ms_solve_sum,
            "upd": ms_upd_sum,
            "cert": ms_cert_sum,
        },
    )
