from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    symmetrize,
)
from polar.rational.ops import (
    cert_bound_trace_logdet_stable,
    gram_xtx_chunked_fast,
    apply_right_small_chunked_fast,
)
from polar.rational.dwh import (
    dwh_ell_next,
)
from polar.rational.dwh_stable_solve import (
    dwh_step_matrix_only_stable_solve,
)
from polar.polynomial.express import (
    polar_express_step_bf16_quadratic,
    polar_express_action_chunked,
    polar_express_fro_scale,
)
from polar.schedules import StepSpec
from polar.runner import RunSummary, exact_final_kappa_O

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_business(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec], # Ignored for business runner
    iter_dtype: torch.dtype,
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    jitter_rel: float,
    cert_jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
    stop_on_cert: bool,
) -> RunSummary:
    """
    ULTIMATE WALL-CLOCK PERFORMANCE path.
    1. FP32 DWH (Pre-condition): 2 steps to map 1e-6 -> ~0.1
    2. BF16 Polar Express (Final stretch): pure matmuls at peak hardware speed.
    """
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    X = G_storage.to(dtype=torch.float32)
    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    ms_cert_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    
    # --- PHASE 1: FP32 DWH PRE-CONDITIONING ---
    # We target 2 steps to get sigma_min to ~0.1.
    # ell0 = 1e-6 -> ell1 = 0.025 -> ell2 = 0.64 (theory)
    ell = 1.0 / (G_storage.shape[1] * 1000.0) # Heuristic ell
    for i in range(2):
        # Compute S in FP32 (TF32 enabled)
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, torch.float32))
        ms_gram_sum += ms_gram
        
        # solve_ex in FP32
        ms_solve, (Q_step, shift) = cuda_time_ms(
            lambda: dwh_step_matrix_only_stable_solve(S, ell, jitter_rel)
        )
        ms_solve_sum += ms_solve
        dwh_steps += 1
        
        # Update X
        ms_upd, X = cuda_time_ms(lambda: X @ Q_step)
        ms_upd_sum += ms_upd
        
        ell = dwh_ell_next(ell)

    # --- PHASE 2: BF16 POLAR EXPRESS ---
    # Switch to BF16 for the absolute maximum speed.
    X_bf16 = X.to(dtype=torch.bfloat16)
    
    # Scale X to avoid BF16 overflow/underflow during matmuls
    ms_upd, (X_bf16, fro_scale) = cuda_time_ms(lambda: polar_express_fro_scale(X_bf16))
    ms_upd_sum += ms_upd
    
    # We do a few steps of quadratic PE to finish it off.
    curr_sigma_lo = ell
    curr_sigma_hi = 1.0 / fro_scale # Rough estimate
    
    for _ in range(3):
        # Compute S in BF16 (TF32 enabled)
        ms_gram, S_bf16 = cuda_time_ms(lambda: gram_xtx_chunked_fast(X_bf16, gram_chunk_rows, torch.bfloat16))
        ms_gram_sum += ms_gram
        
        # Generate PE coefficients for the current interval
        pe_coeffs = polar_express_step_bf16_quadratic(curr_sigma_lo, curr_sigma_hi)
        
        # Action step: pure BF16 matmuls
        ms_solve, (X_bf16, _) = cuda_time_ms(
            lambda: polar_express_action_chunked(X_bf16, S_bf16, pe_coeffs, rhs_chunk_rows, torch.bfloat16)
        )
        ms_solve_sum += ms_solve
        zolo_steps += 1 # Counting PE as ZOLO-like polynomial
        
        curr_sigma_lo = pe_coeffs.pred_sigma_min
        curr_sigma_hi = pe_coeffs.pred_sigma_max
        if curr_sigma_lo >= 0.999:
            break

    # Cast back to iter_dtype
    X = X_bf16.to(dtype=iter_dtype)

    # --- FINALIZATION ---
    # Final certificate (always in FP64 internally but on FP32 S)
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, torch.float32))
    ms_gram_sum += ms_gram
    ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
        lambda: cert_bound_trace_logdet_stable(S, cert_jitter_rel)
    )
    ms_cert_sum += ms_cert

    # Verification
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, gram_chunk_rows, exact_verify_device)
    )
    
    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
    return RunSummary(
        success=bool(final_kO_exact <= target_kappa_O),
        final_kO_exact=float(final_kO_exact),
        final_kO_cert=float(kO_cert),
        steps=dwh_steps + zolo_steps,
        dwh_steps=dwh_steps,
        zolo_steps=zolo_steps,
        guards=guards,
        fallbacks=fallbacks,
        last_step_kind="PE_BF16",
        ms_gram=ms_gram_sum,
        ms_solve=ms_solve_sum,
        ms_upd=ms_upd_sum,
        ms_cert=ms_cert_sum,
        ms_total_timed=ms_total,
        ms_exact_verify=ms_exact_verify,
    )
