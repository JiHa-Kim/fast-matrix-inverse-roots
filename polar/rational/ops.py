from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import (
    symmetrize,
    safe_exp,
    acosh_exp,
    chol_with_jitter_fp64,
)

Tensor = torch.Tensor

@torch.no_grad()
def cert_bound_trace_logdet_stable(S: Tensor, jitter_rel: float) -> Tuple[float, float]:
    """
    More robust certificate calculation that falls back to eigvalsh if Cholesky fails.
    Useful for lower precision matrices.
    """
    S_work = symmetrize(S.to(torch.float64))
    n = S_work.shape[0]

    try:
        L, shift = chol_with_jitter_fp64(S_work, jitter_rel=jitter_rel)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()
    except RuntimeError:
        # Fallback to eigvalsh if Cholesky fails even after jitter
        evals = torch.linalg.eigvalsh(S_work)
        # Ensure all eigenvalues are positive for logdet calculation
        evals = torch.clamp(evals, min=1e-300)
        logdet = torch.log(evals).sum().item()
        shift = 0.0 # We didn't use shift for eigvalsh

    a = max(float((torch.trace(S_work) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    return float(safe_exp(eta_ub)), float(shift)


@torch.no_grad()
def apply_right_small_chunked_fast(
    X: Tensor, U: Tensor, rhs_chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    """
    Lower-precision version of apply_right_small_chunked.
    Optimized for speed using TF32.
    """
    m, n = X.shape
    X_next = torch.empty((m, n), device=X.device, dtype=out_dtype)
    
    # Enable TF32 for the matmul if we are in float32
    orig_precision = torch.get_float32_matmul_precision()
    if X.dtype == torch.float32 or U.dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
        
    try:
        # Use the requested out_dtype for the matmul to gain speed
        U_work = U.to(dtype=out_dtype)

        for i in range(0, m, rhs_chunk_rows):
            Xi = X[i : i + rhs_chunk_rows].to(dtype=out_dtype)
            # This matmul will use TF32 on modern GPUs
            Zi = Xi @ U_work
            X_next[i : i + rhs_chunk_rows] = Zi
    finally:
        torch.set_float32_matmul_precision(orig_precision)

    return X_next


@torch.no_grad()
def gram_xtx_chunked_fast(X: Tensor, chunk_rows: int, accum_dtype: torch.dtype) -> Tensor:
    """
    Chunked Gram matrix calculation for the fused fast runners.
    """
    m, n = X.shape
    S = torch.zeros((n, n), device=X.device, dtype=accum_dtype)

    orig_precision = torch.get_float32_matmul_precision()
    if X.dtype == torch.float32 or accum_dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    try:
        for i in range(0, m, chunk_rows):
            Xi = X[i : i + chunk_rows].to(dtype=accum_dtype)
            S.addmm_(Xi.mT, Xi)
        return symmetrize(S)
    finally:
        torch.set_float32_matmul_precision(orig_precision)


@torch.no_grad()
def gram_xtx_fast(X: Tensor, accum_dtype: torch.dtype) -> Tensor:
    """
    Ultra-fast Gram matrix calculation. No chunking.
    """
    orig_precision = torch.get_float32_matmul_precision()
    if X.dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    try:
        # Full matmul for peak occupancy
        S = X.mT @ X
        return symmetrize(S.to(dtype=accum_dtype))
    finally:
        torch.set_float32_matmul_precision(orig_precision)


@torch.no_grad()
def apply_right_fast(X: Tensor, Q: Tensor, out_dtype: torch.dtype) -> Tensor:
    """
    Ultra-fast matrix update. No chunking.
    """
    orig_precision = torch.get_float32_matmul_precision()
    if X.dtype == torch.float32 or Q.dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    try:
        # Full matmul for peak occupancy
        return (X @ Q).to(dtype=out_dtype)
    finally:
        torch.set_float32_matmul_precision(orig_precision)


@torch.no_grad()
def exact_final_kappa_O_fast(X: Tensor) -> float:
    """
    Faster exact verification using pure FP32/TF32 if acceptable.
    Actually, for 'exact' we should stay in FP64 but we can use no-chunking.
    """
    X_64 = X.to(torch.float64)
    S = X_64.mT @ X_64
    evals = torch.linalg.eigvalsh(S)
    kappa = float(torch.sqrt(evals[-1] / evals[0].clamp_min(1e-30)).item())
    return kappa
