#!/usr/bin/env python3
# action_invquarter_coeffsolve_checked.py
#
# Correctness-first rational-action baseline for computing G P^{-1/4}
# without explicitly forming P^{-1/4} in the main iteration.
#
# Compared to the earlier file, this version adds an actual correctness suite:
#   - exact oracle for P^{-1/4} via eigendecomposition when n is small enough,
#   - exact action error ||Y - G P^{-1/4}|| / ||G P^{-1/4}||,
#   - exact small-side root error ||X - P^{-1/4}|| / ||P^{-1/4}|| when we track X,
#   - replay check ||Y - G X|| / ||G X|| to separate action-storage error from root error.
#
# The main algorithmic iterate is unchanged:
#   M_k = X_k^4 P,
#   Y_k = G X_k,
#   X_{k+1} = X_k q_k(M_k),
#   Y_{k+1} = Y_k q_k(M_k),
#   M_{k+1} = q_k(M_k)^4 M_k.

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy import optimize

Tensor = torch.Tensor


# ----------------------------- utilities ------------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def pct(xs: List[float], p: float) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return float("nan")
    ys.sort()
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        t0 = time.time()
        out = fn()
        return 1000.0 * (time.time() - t0), out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_fro(A: Tensor, B: Tensor) -> float:
    num = float(torch.linalg.matrix_norm(A - B, ord="fro").item())
    den = max(float(torch.linalg.matrix_norm(B, ord="fro").item()), 1e-300)
    return float(num / den)


def rel_spec(A: Tensor, B: Tensor) -> float:
    num = float(torch.linalg.matrix_norm(A - B, ord=2).item())
    den = max(float(torch.linalg.matrix_norm(B, ord=2).item()), 1e-300)
    return float(num / den)


# ----------------------------- fp64 SPD ops ---------------------------------


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
    I = torch.eye(n, device=A.device, dtype=torch.float64)

    scale = float((torch.trace(A).abs() / max(n, 1)).item())
    base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)

    delta = 0.0
    for _ in range(max_tries):
        At = A if delta == 0.0 else (A + delta * I)
        L, info = torch.linalg.cholesky_ex(At)
        if int(info.item()) == 0:
            return L, float(delta)
        delta = base if delta == 0.0 else 2.0 * delta

    raise RuntimeError("Cholesky failed even after jitter escalation")


@torch.no_grad()
def make_spd_honest_fp64(P: Tensor, jitter_rel: float) -> Tuple[Tensor, float]:
    P = symmetrize(P.to(torch.float64))
    _, shift = chol_with_jitter_fp64(P, jitter_rel=jitter_rel)
    if shift > 0.0:
        n = P.shape[0]
        I = torch.eye(n, device=P.device, dtype=torch.float64)
        P = symmetrize(P + shift * I)
    return P, float(shift)


@torch.no_grad()
def init_spectrum_exact_fp64(P: Tensor) -> Tuple[float, float]:
    evals = torch.linalg.eigvalsh(symmetrize(P.to(torch.float64)))
    lam_min = max(float(evals[0].item()), 1e-300)
    lam_max = max(float(evals[-1].item()), lam_min)
    return float(lam_min), float(lam_max)


# ----------------------------- direct coefficient solve ---------------------


@dataclasses.dataclass
class InvQuarterCoeffSolve:
    ds_norm: np.ndarray
    residues_norm: np.ndarray
    alpha0_norm: float
    gmin_norm: float
    gmax_norm: float
    max_rel_fit: float
    max_next_resid_pred: float
    nit: int
    success: bool
    log_shifts: np.ndarray


@dataclasses.dataclass
class ScaledInvQuarterCoeffs:
    betas: List[float]
    gammas: List[float]
    gamma0: float
    gmin_next: float
    gmax_next: float
    max_rel_fit: float
    max_next_resid_pred: float
    nit: int
    success: bool
    log_shifts: np.ndarray


@torch.no_grad()
def solve_invquarter_coeffs_normalized(
    ell: float,
    r: int,
    n_grid: int,
    maxiter: int,
    init_log_shifts: Optional[np.ndarray] = None,
) -> InvQuarterCoeffSolve:
    ell = float(min(max(ell, 1e-12), 1.0))

    if ell >= 1.0 - 1e-12:
        return InvQuarterCoeffSolve(
            ds_norm=np.array([1.0] * r, dtype=np.float64),
            residues_norm=np.zeros((r,), dtype=np.float64),
            alpha0_norm=1.0,
            gmin_norm=1.0,
            gmax_norm=1.0,
            max_rel_fit=0.0,
            max_next_resid_pred=0.0,
            nit=0,
            success=True,
            log_shifts=np.zeros((r,), dtype=np.float64),
        )

    grid_geom = np.geomspace(ell, 1.0, n_grid)
    grid_lin = np.linspace(ell, 1.0, n_grid)
    ts = np.unique(np.concatenate([grid_geom, grid_lin])).astype(np.float64)

    w = np.ones_like(ts)
    edge = max(8, n_grid // 20)
    w[:edge] *= 5.0
    w[-edge:] *= 5.0

    pdeg = r

    def fit_from_logshifts(logd: np.ndarray):
        ds = np.exp(np.sort(logd.astype(np.float64)))

        D = np.ones_like(ts)
        for d in ds:
            D *= ts + d

        A = np.stack([(ts**j) * (ts**0.25) / D for j in range(pdeg + 1)], axis=1)
        Aw = A * w[:, None]
        bw = np.ones_like(ts) * w
        ncoef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

        N = np.zeros_like(ts)
        for j in range(pdeg + 1):
            N += ncoef[j] * (ts**j)
        q = N / D

        rel = q * (ts**0.25) - 1.0
        g = ts * (q**4)

        penalty = 0.0
        if np.any(~np.isfinite(q)) or np.any(~np.isfinite(g)):
            penalty += 1e6
        if np.any(q <= 0.0):
            penalty += 1e6
        if np.any(g <= 0.0):
            penalty += 1e6

        return ds, ncoef, rel, g, penalty

    def objective(logd: np.ndarray) -> float:
        _, _, rel, g, penalty = fit_from_logshifts(logd)
        if penalty > 0.0:
            return float(penalty)
        return float(max(np.max(np.abs(rel)), np.max(np.abs(g - 1.0))))

    if init_log_shifts is None or len(init_log_shifts) != r:
        x0 = np.log(
            np.array([ell ** ((j + 1) / (r + 1)) for j in range(r)], dtype=np.float64)
        )
    else:
        x0 = np.array(init_log_shifts, dtype=np.float64)

    opt = optimize.minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-7, "fatol": 1e-7, "disp": False},
    )

    ds, ncoef, rel, g, _ = fit_from_logshifts(opt.x)

    alpha0 = float(ncoef[-1])
    residues = []
    for j, d in enumerate(ds):
        n_at = 0.0
        for k in range(len(ncoef)):
            n_at += float(ncoef[k]) * ((-float(d)) ** k)
        dprime_at = 1.0
        for k, dk in enumerate(ds):
            if k != j:
                dprime_at *= -float(d) + float(dk)
        residues.append(float(n_at / dprime_at))

    return InvQuarterCoeffSolve(
        ds_norm=np.array(ds, dtype=np.float64),
        residues_norm=np.array(residues, dtype=np.float64),
        alpha0_norm=alpha0,
        gmin_norm=float(np.min(g)),
        gmax_norm=float(np.max(g)),
        max_rel_fit=float(np.max(np.abs(rel))),
        max_next_resid_pred=float(np.max(np.abs(g - 1.0))),
        nit=int(getattr(opt, "nit", 0)),
        success=bool(opt.success),
        log_shifts=np.log(np.array(ds, dtype=np.float64)),
    )


@torch.no_grad()
def scale_invquarter_coeffs_to_interval(
    coeffs: InvQuarterCoeffSolve, lam_hi: float
) -> ScaledInvQuarterCoeffs:
    lam_hi = max(float(lam_hi), 1e-300)
    betas = [float(lam_hi * d) for d in coeffs.ds_norm.tolist()]
    gammas = [float((lam_hi**0.75) * a) for a in coeffs.residues_norm.tolist()]
    gamma0 = float((lam_hi ** (-0.25)) * coeffs.alpha0_norm)

    return ScaledInvQuarterCoeffs(
        betas=betas,
        gammas=gammas,
        gamma0=gamma0,
        gmin_next=float(coeffs.gmin_norm),
        gmax_next=float(coeffs.gmax_norm),
        max_rel_fit=float(coeffs.max_rel_fit),
        max_next_resid_pred=float(coeffs.max_next_resid_pred),
        nit=int(coeffs.nit),
        success=bool(coeffs.success),
        log_shifts=np.array(coeffs.log_shifts, dtype=np.float64),
    )


# ----------------------------- shifted-solve action -------------------------


@torch.no_grad()
def build_q_from_scaled_pf(
    M: Tensor,
    scaled_coeffs: ScaledInvQuarterCoeffs,
    jitter_rel: float,
) -> Tuple[Tensor, List[Tensor], int, float]:
    M = symmetrize(M.to(torch.float64))
    n = M.shape[0]
    I = torch.eye(n, device=M.device, dtype=torch.float64)

    Q = scaled_coeffs.gamma0 * I
    chol_factors: List[Tensor] = []
    guards = 0
    max_shift = 0.0

    for beta, gamma in zip(scaled_coeffs.betas, scaled_coeffs.gammas):
        A = symmetrize(M + float(beta) * I)
        L, shift = chol_with_jitter_fp64(A, jitter_rel=jitter_rel)
        S = torch.cholesky_solve(I, L)
        Q = Q + float(gamma) * S
        chol_factors.append(L)
        guards += int(shift > 0.0)
        max_shift = max(max_shift, float(shift))

    return symmetrize(Q), chol_factors, guards, max_shift


@torch.no_grad()
def apply_pf_action_chunked(
    Y: Tensor,
    chol_factors: List[Tensor],
    scaled_coeffs: ScaledInvQuarterCoeffs,
    rhs_chunk_rows: int,
    out_dtype: torch.dtype,
) -> Tensor:
    m, n = Y.shape
    out = torch.empty((m, n), device=Y.device, dtype=out_dtype)

    for i in range(0, m, rhs_chunk_rows):
        Yi = Y[i : i + rhs_chunk_rows].float().to(torch.float64)
        Zi = scaled_coeffs.gamma0 * Yi
        for L, gamma in zip(chol_factors, scaled_coeffs.gammas):
            Ti_t = torch.cholesky_solve(Yi.T.contiguous(), L)
            Zi = Zi + float(gamma) * Ti_t.T
        out[i : i + rhs_chunk_rows] = Zi.to(out_dtype)

    return out


# ----------------------------- certification --------------------------------


@torch.no_grad()
def cert_from_m(
    M: Tensor,
    cert_mode: str,
    exact_threshold: int,
    cert_jitter_rel: float,
) -> Tuple[float, float, float, float]:
    """
    Quarter-root action certificate for M = X^4 P.

    Returns:
      action_rel_cert,
      action_rel_exact_or_nan,
      resid_M_cert,
      shift_used_for_cert

    Exact mode computes
      max(|lam^(1/4) - 1|)
    over the eigenvalues lam of M.

    Bound mode first upper-bounds ||M - I||_2 by ||M - I||_F = eta.
    If eta < 1, then the spectrum of M lies in [1-eta, 1+eta], hence
      ||M^(1/4) - I||_2 <= max(1 - (1-eta)^(1/4), (1+eta)^(1/4) - 1).
    If eta >= 1, the lower endpoint bound is no longer informative, so we
    return infinity as the action certificate and keep resid_M_cert for
    diagnostics.
    """
    M = symmetrize(M.to(torch.float64))
    n = M.shape[0]
    I = torch.eye(n, device=M.device, dtype=torch.float64)

    use_exact = (cert_mode == "exact") or (cert_mode == "auto" and n <= exact_threshold)

    if use_exact:
        evals = torch.linalg.eigvalsh(M)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        action_rel = max(1.0 - lam_min**0.25, lam_max**0.25 - 1.0)
        resid = float(max(abs(lam_min - 1.0), abs(lam_max - 1.0)))
        return float(action_rel), float(action_rel), float(resid), 0.0

    _, shift = chol_with_jitter_fp64(M, jitter_rel=cert_jitter_rel)
    resid_ub = float(torch.linalg.matrix_norm(M - I, ord="fro").item())
    if resid_ub >= 1.0:
        action_rel_ub = float("inf")
    else:
        action_rel_ub = max(
            1.0 - (1.0 - resid_ub) ** 0.25, (1.0 + resid_ub) ** 0.25 - 1.0
        )
    return float(action_rel_ub), float("nan"), float(resid_ub), float(shift)


# ----------------------------- synthetic data -------------------------------


@dataclasses.dataclass
class SpectrumSpec:
    name: str
    eigs: Tensor


def make_spd_from_eigs(
    eigs: Tensor, seed: int, device: str, storage_dtype: torch.dtype
) -> Tensor:
    n = int(eigs.numel())
    seed_all(seed)
    Q, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=torch.float64), mode="reduced"
    )
    P = (Q * eigs.to(device=device, dtype=torch.float64)) @ Q.T
    return symmetrize(P).to(dtype=storage_dtype)


def make_random_G(
    m: int, n: int, seed: int, device: str, storage_dtype: torch.dtype
) -> Tensor:
    seed_all(seed)
    G = torch.randn(m, n, device=device, dtype=torch.float32)
    return G.to(dtype=storage_dtype)


def make_eig_bank(
    n: int, kappa_P: float, bank_size: int, seed: int
) -> List[SpectrumSpec]:
    lam_max = 1.0
    lam_min = 1.0 / float(kappa_P)
    out: List[SpectrumSpec] = []

    def add(name: str, eigs: Tensor) -> None:
        eigs = eigs.to(torch.float64).clone()
        eigs[0] = lam_max
        eigs[-1] = lam_min
        eigs = torch.clamp(eigs, min=lam_min, max=lam_max)
        eigs, _ = torch.sort(eigs, descending=True)
        out.append(SpectrumSpec(name=name, eigs=eigs))

    add(
        "logspace",
        torch.logspace(0.0, math.log10(lam_min), n, base=10.0, dtype=torch.float64),
    )

    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs1 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (t**p)
        logs2 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (
            1.0 - (1.0 - t) ** p
        )
        add(f"power_front_p{p:g}", torch.exp(logs1))
        add(f"power_back_p{p:g}", torch.exp(logs2))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        d = torch.full((n,), lam_min, dtype=torch.float64)
        d[:r] = lam_max
        add(f"step_frac_{frac:.4g}", d)

    for frac in [0.02, 0.05, 0.1]:
        r1 = max(1, min(n - 2, int(round(frac * n))))
        r2 = max(r1 + 1, min(n - 1, int(round((0.5 + 0.5 * frac) * n))))
        d = torch.full((n,), lam_min, dtype=torch.float64)
        d[:r1] = lam_max
        d[r1:r2] = math.sqrt(lam_min)
        add(f"three_level_frac_{frac:.3g}", d)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(lam_min) * x for x in u], dtype=torch.float64)
        d = torch.exp(logs)
        add(f"random_monotone_{len(out)}", d)

    return out[:bank_size]


def suite_shapes_kimi_glm5() -> List[Tuple[int, int]]:
    return [
        (2048, 256),
        (4096, 256),
        (8192, 256),
        (8192, 1024),
        (16384, 1024),
        (8192, 2048),
        (16384, 2048),
        (28672, 4096),
        (28672, 7168),
        (32768, 8192),
    ]


# ----------------------------- oracle ---------------------------------------


@dataclasses.dataclass
class OracleSummary:
    computed: bool
    root_rel_fro: float
    root_rel_spec: float
    action_rel_fro: float
    action_rel_spec: float
    replay_rel_fro: float
    replay_rel_spec: float
    quarter_resid_exact: float
    ms_oracle: float


@torch.no_grad()
def exact_invquarter_fp64(P: Tensor) -> Tensor:
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    return symmetrize((U * evals.pow(-0.25)) @ U.T)


@torch.no_grad()
def exact_p_eighth_fp64(P: Tensor) -> Tensor:
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    return symmetrize((U * evals.pow(0.125)) @ U.T)


@torch.no_grad()
def compute_oracle(
    G_storage: Tensor,
    P_honest: Tensor,
    Y_approx: Tensor,
    X_track: Optional[Tensor],
) -> OracleSummary:
    G64 = G_storage.float().to(torch.float64)
    P64 = symmetrize(P_honest.to(torch.float64))
    n = P64.shape[0]
    I = torch.eye(n, device=P64.device, dtype=torch.float64)

    X_exact = exact_invquarter_fp64(P64)
    P_eighth = exact_p_eighth_fp64(P64)
    Y_exact = G64 @ X_exact

    if X_track is None:
        X_track = X_exact.clone()

    X_track = symmetrize(X_track.to(torch.float64))
    Y_track = G64 @ X_track
    Y_approx64 = Y_approx.float().to(torch.float64)

    stable_resid = symmetrize(P_eighth @ X_track @ P_eighth) - I
    quarter_resid_exact = float(torch.linalg.matrix_norm(stable_resid, ord=2).item())

    return OracleSummary(
        computed=True,
        root_rel_fro=rel_fro(X_track, X_exact),
        root_rel_spec=rel_spec(X_track, X_exact),
        action_rel_fro=rel_fro(Y_approx64, Y_exact),
        action_rel_spec=rel_spec(Y_approx64, Y_exact),
        replay_rel_fro=rel_fro(Y_approx64, Y_track),
        replay_rel_spec=rel_spec(Y_approx64, Y_track),
        quarter_resid_exact=quarter_resid_exact,
        ms_oracle=0.0,
    )


# ----------------------------- run core -------------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_action_rel_cert: float
    final_action_rel_exact: float
    final_resid_M_cert: float
    pred_lo: float
    pred_hi: float
    coeff_max_rel_fit: float
    coeff_max_next_resid_pred: float
    steps: int
    guards: int
    coeff_successes: int
    coeff_failures: int
    coeff_nit_last: int
    ms_init: float
    ms_coeff: float
    ms_small: float
    ms_apply: float
    ms_cert: float
    ms_oracle: float
    ms_total: float
    oracle: OracleSummary


@torch.no_grad()
def _init_small_side(P_storage: Tensor, solve_jitter_rel: float):
    P, spd_shift = make_spd_honest_fp64(P_storage, jitter_rel=solve_jitter_rel)
    lam_min, lam_max = init_spectrum_exact_fp64(P)
    return P, spd_shift, lam_min, lam_max


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    P_storage: Tensor,
    target_action_rel: float,
    max_steps: int,
    iter_dtype: torch.dtype,
    cert_mode: str,
    exact_threshold: int,
    rhs_chunk_rows: int,
    solve_jitter_rel: float,
    cert_jitter_rel: float,
    coeff_r: int,
    coeff_grid: int,
    coeff_maxiter: int,
    oracle_mode: str,
    oracle_n_max: int,
) -> Tuple[Tensor, RunSummary]:
    ms_init, (P, spd_shift, lam_min, lam_max) = cuda_time_ms(
        lambda: _init_small_side(P_storage, solve_jitter_rel)
    )
    guards = int(spd_shift > 0.0)

    scale = lam_max**0.25
    M = symmetrize(P / lam_max)
    Y = G_storage.to(dtype=iter_dtype) / scale

    do_oracle = (oracle_mode == "on") or (
        oracle_mode == "auto" and P.shape[0] <= oracle_n_max
    )
    X_track = None
    if do_oracle:
        n = P.shape[0]
        X_track = torch.eye(n, device=P.device, dtype=torch.float64) / scale

    pred_lo = max(lam_min / lam_max, 1e-300)
    pred_hi = 1.0
    warm_log_shifts: Optional[np.ndarray] = None

    ms_coeff_sum = 0.0
    ms_small_sum = 0.0
    ms_apply_sum = 0.0
    ms_cert_sum = 0.0
    ms_oracle = 0.0

    final_action_rel_cert = float("inf")
    final_action_rel_exact = float("nan")
    final_resid_M_cert = float("inf")
    coeff_max_rel_fit = float("inf")
    coeff_max_next_resid_pred = float("inf")
    coeff_successes = 0
    coeff_failures = 0
    coeff_nit_last = 0

    for it in range(1, max_steps + 1):
        ms_coeff, coeff_solve = cuda_time_ms(
            lambda: solve_invquarter_coeffs_normalized(
                ell=max(pred_lo / max(pred_hi, 1e-300), 1e-300),
                r=coeff_r,
                n_grid=coeff_grid,
                maxiter=coeff_maxiter,
                init_log_shifts=warm_log_shifts,
            )
        )
        ms_coeff_sum += ms_coeff
        warm_log_shifts = coeff_solve.log_shifts
        coeff_successes += int(coeff_solve.success)
        coeff_failures += int(not coeff_solve.success)
        coeff_nit_last = int(coeff_solve.nit)

        scaled_coeffs = scale_invquarter_coeffs_to_interval(coeff_solve, pred_hi)
        coeff_max_rel_fit = float(scaled_coeffs.max_rel_fit)
        coeff_max_next_resid_pred = float(scaled_coeffs.max_next_resid_pred)

        ms_small, (Q, chol_factors, more_guards, _) = cuda_time_ms(
            lambda: build_q_from_scaled_pf(
                M=M, scaled_coeffs=scaled_coeffs, jitter_rel=solve_jitter_rel
            )
        )
        ms_small_sum += ms_small
        guards += int(more_guards)

        if X_track is not None:
            X_track = symmetrize(X_track @ Q)

        Q2 = symmetrize(Q @ Q)
        Q4 = symmetrize(Q2 @ Q2)
        M = symmetrize(Q4 @ M)

        ms_apply, Y = cuda_time_ms(
            lambda: apply_pf_action_chunked(
                Y=Y,
                chol_factors=chol_factors,
                scaled_coeffs=scaled_coeffs,
                rhs_chunk_rows=rhs_chunk_rows,
                out_dtype=iter_dtype,
            )
        )
        ms_apply_sum += ms_apply

        pred_lo = float(scaled_coeffs.gmin_next)
        pred_hi = float(scaled_coeffs.gmax_next)

        ms_cert, (action_rel_cert, action_rel_exact, resid_M_cert, cert_shift) = (
            cuda_time_ms(
                lambda: cert_from_m(
                    M=M,
                    cert_mode=cert_mode,
                    exact_threshold=exact_threshold,
                    cert_jitter_rel=cert_jitter_rel,
                )
            )
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)

        final_action_rel_cert = float(action_rel_cert)
        final_action_rel_exact = float(action_rel_exact)
        final_resid_M_cert = float(resid_M_cert)

        if final_action_rel_cert <= target_action_rel:
            break

    oracle = OracleSummary(
        computed=False,
        root_rel_fro=float("nan"),
        root_rel_spec=float("nan"),
        action_rel_fro=float("nan"),
        action_rel_spec=float("nan"),
        replay_rel_fro=float("nan"),
        replay_rel_spec=float("nan"),
        quarter_resid_exact=float("nan"),
        ms_oracle=0.0,
    )

    if do_oracle:
        ms_oracle, oracle = cuda_time_ms(
            lambda: compute_oracle(
                G_storage=G_storage, P_honest=P, Y_approx=Y, X_track=X_track
            )
        )
        oracle.ms_oracle = ms_oracle

    ms_total = (
        ms_init + ms_coeff_sum + ms_small_sum + ms_apply_sum + ms_cert_sum + ms_oracle
    )
    return Y, RunSummary(
        success=(final_action_rel_cert <= target_action_rel),
        final_action_rel_cert=final_action_rel_cert,
        final_action_rel_exact=final_action_rel_exact,
        final_resid_M_cert=final_resid_M_cert,
        pred_lo=float(pred_lo),
        pred_hi=float(pred_hi),
        coeff_max_rel_fit=coeff_max_rel_fit,
        coeff_max_next_resid_pred=coeff_max_next_resid_pred,
        steps=it,
        guards=guards,
        coeff_successes=coeff_successes,
        coeff_failures=coeff_failures,
        coeff_nit_last=coeff_nit_last,
        ms_init=ms_init,
        ms_coeff=ms_coeff_sum,
        ms_small=ms_small_sum,
        ms_apply=ms_apply_sum,
        ms_cert=ms_cert_sum,
        ms_oracle=ms_oracle,
        ms_total=ms_total,
        oracle=oracle,
    )


# ----------------------------- CLI ------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--kappa_P", type=float, default=1e7)
    ap.add_argument("--target_action_rel", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=4)

    ap.add_argument("--input_dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16"], default="float32")

    ap.add_argument("--cert_mode", choices=["auto", "exact", "bound"], default="auto")
    ap.add_argument("--exact_threshold", type=int, default=1024)
    ap.add_argument("--rhs_chunk_rows", type=int, default=2048)
    ap.add_argument("--solve_jitter_rel", type=float, default=1e-15)
    ap.add_argument("--cert_jitter_rel", type=float, default=1e-15)

    ap.add_argument("--coeff_r", type=int, default=4)
    ap.add_argument("--coeff_grid", type=int, default=120)
    ap.add_argument("--coeff_maxiter", type=int, default=800)

    ap.add_argument("--oracle_mode", choices=["off", "auto", "on"], default="on")
    ap.add_argument("--oracle_n_max", type=int, default=2048)

    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    input_dtype = torch.float32 if args.input_dtype == "float32" else torch.bfloat16
    iter_dtype = torch.float32 if args.iter_dtype == "float32" else torch.bfloat16

    print(
        f"device={args.device}  mode={args.mode}  target=G P^(-1/4)  "
        f"kappa_P<={args.kappa_P:.3g}  target_action_rel<={args.target_action_rel:.3g}"
    )
    print(
        "knobs: "
        f"max_steps={args.max_steps} input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"cert_mode={args.cert_mode} exact_threshold={args.exact_threshold} "
        f"rhs_chunk_rows={args.rhs_chunk_rows} coeff_r={args.coeff_r} "
        f"coeff_grid={args.coeff_grid} coeff_maxiter={args.coeff_maxiter} "
        f"oracle_mode={args.oracle_mode} oracle_n_max={args.oracle_n_max} "
        f"solve_jitter_rel={args.solve_jitter_rel:g} cert_jitter_rel={args.cert_jitter_rel:g}"
    )

    def make_case(m: int, n: int, case_seed: int) -> Tuple[Tensor, Tensor]:
        bank = make_eig_bank(
            n, args.kappa_P, bank_size=max(args.bank_size, 24), seed=args.seed + 17 * n
        )
        spec = bank[case_seed % len(bank)]
        P = make_spd_from_eigs(
            eigs=spec.eigs,
            seed=case_seed,
            device=args.device,
            storage_dtype=input_dtype,
        )
        G = make_random_G(
            m=m, n=n, seed=case_seed + 1, device=args.device, storage_dtype=input_dtype
        )
        return G, P

    def run_case(G: Tensor, P: Tensor) -> RunSummary:
        _, res = run_one_case(
            G_storage=G,
            P_storage=P,
            target_action_rel=args.target_action_rel,
            max_steps=args.max_steps,
            iter_dtype=iter_dtype,
            cert_mode=args.cert_mode,
            exact_threshold=args.exact_threshold,
            rhs_chunk_rows=args.rhs_chunk_rows,
            solve_jitter_rel=args.solve_jitter_rel,
            cert_jitter_rel=args.cert_jitter_rel,
            coeff_r=args.coeff_r,
            coeff_grid=args.coeff_grid,
            coeff_maxiter=args.coeff_maxiter,
            oracle_mode=args.oracle_mode,
            oracle_n_max=args.oracle_n_max,
        )
        return res

    def print_oracle(res: RunSummary) -> None:
        if not res.oracle.computed:
            print("  oracle: skipped")
            return
        print(
            f"  oracle action rel_fro={res.oracle.action_rel_fro:.6g} rel_spec={res.oracle.action_rel_spec:.6g} "
            f"replay_rel_fro={res.oracle.replay_rel_fro:.6g} replay_rel_spec={res.oracle.replay_rel_spec:.6g}"
        )
        print(
            f"  oracle root   rel_fro={res.oracle.root_rel_fro:.6g} rel_spec={res.oracle.root_rel_spec:.6g} "
            f"quarter_resid_exact={res.oracle.quarter_resid_exact:.6g}"
        )

    if args.mode == "demo":
        G, P = make_case(args.m, args.n, args.seed)
        res = run_case(G, P)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} "
            f"action_rel_cert={res.final_action_rel_cert:.6g} "
            f"exact={res.final_action_rel_exact:.6g} "
            f"resid(M)_cert={res.final_resid_M_cert:.6g} "
            f"pred_interval=[{res.pred_lo:.6g}, {res.pred_hi:.6g}] "
            f"coeff_fit={res.coeff_max_rel_fit:.6g} coeff_next_resid_pred={res.coeff_max_next_resid_pred:.6g} "
            f"steps={res.steps} guards={res.guards} coeff_nit_last={res.coeff_nit_last}"
        )
        print(f"  coeff success/fail={res.coeff_successes}/{res.coeff_failures}")
        print_oracle(res)
        print(
            f"  ms total={res.ms_total:.3f} "
            f"(init={res.ms_init:.3f} coeff={res.ms_coeff:.3f} small={res.ms_small:.3f} "
            f"apply={res.ms_apply:.3f} cert={res.ms_cert:.3f} oracle={res.ms_oracle:.3f})"
        )
        return

    def summarize(results: List[RunSummary], target_cases: int) -> None:
        action_cert = [r.final_action_rel_cert for r in results]
        action_exact = [r.final_action_rel_exact for r in results]
        resid_M = [r.final_resid_M_cert for r in results]
        coeff_fit = [r.coeff_max_rel_fit for r in results]
        coeff_pred = [r.coeff_max_next_resid_pred for r in results]
        steps = [r.steps for r in results]
        guards = [r.guards for r in results]
        coeff_failures = [r.coeff_failures for r in results]
        ms_total = [r.ms_total for r in results]
        ms_oracle = [r.ms_oracle for r in results]
        action_rel_fro = [r.oracle.action_rel_fro for r in results if r.oracle.computed]
        action_rel_spec = [
            r.oracle.action_rel_spec for r in results if r.oracle.computed
        ]
        replay_rel_fro = [r.oracle.replay_rel_fro for r in results if r.oracle.computed]
        root_rel_fro = [r.oracle.root_rel_fro for r in results if r.oracle.computed]
        quarter_resid = [
            r.oracle.quarter_resid_exact for r in results if r.oracle.computed
        ]

        print(
            f"  success <= target action rel: {sum(int(r.success) for r in results)}/{target_cases}"
        )
        print(
            f"  action rel cert median: {pct(action_cert, 0.5):.6g}  p90: {pct(action_cert, 0.9):.6g}"
        )
        if any(math.isfinite(x) for x in action_exact):
            print(
                f"  action rel exact median: {pct(action_exact, 0.5):.6g}  p90: {pct(action_exact, 0.9):.6g}"
            )
        print(
            f"  resid(M)_cert median: {pct(resid_M, 0.5):.6g}  p90: {pct(resid_M, 0.9):.6g}"
        )
        print(
            f"  coeff fit median: {pct(coeff_fit, 0.5):.6g}  p90: {pct(coeff_fit, 0.9):.6g}"
        )
        print(
            f"  coeff next-resid pred median: {pct(coeff_pred, 0.5):.6g}  p90: {pct(coeff_pred, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        print(
            f"  coeff failures median: {pct(coeff_failures, 0.5):.6g}  p90: {pct(coeff_failures, 0.9):.6g}"
        )
        if action_rel_fro:
            print(
                f"  oracle action rel_fro median: {pct(action_rel_fro, 0.5):.6g}  p90: {pct(action_rel_fro, 0.9):.6g}"
            )
            print(
                f"  oracle action rel_spec median: {pct(action_rel_spec, 0.5):.6g}  p90: {pct(action_rel_spec, 0.9):.6g}"
            )
            print(
                f"  oracle replay rel_fro median: {pct(replay_rel_fro, 0.5):.6g}  p90: {pct(replay_rel_fro, 0.9):.6g}"
            )
            print(
                f"  oracle root rel_fro median: {pct(root_rel_fro, 0.5):.6g}  p90: {pct(root_rel_fro, 0.9):.6g}"
            )
            print(
                f"  oracle quarter resid median: {pct(quarter_resid, 0.5):.6g}  p90: {pct(quarter_resid, 0.9):.6g}"
            )
        else:
            print("  oracle metrics: skipped on all cases")
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        if any(math.isfinite(x) for x in ms_oracle):
            print(
                f"  ms oracle median: {pct(ms_oracle, 0.5):.3f}  p90: {pct(ms_oracle, 0.9):.3f}"
            )

    if args.mode == "bank":
        results: List[RunSummary] = []
        for i in range(args.bank_size):
            try:
                G, P = make_case(args.m, args.n, args.seed + 1000 + i)
                results.append(run_case(G, P))
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        print("")
        print(f"bank summary (N={len(results)}):")
        summarize(results, len(results))
        return

    shapes = (
        suite_shapes_kimi_glm5()
        if args.suite_shapes == "kimi_glm5"
        else [(args.m, args.n)]
    )
    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        results: List[RunSummary] = []
        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G, P = make_case(m, n, args.seed + 10000 + i)
                results.append(run_case(G, P))
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        dt = time.time() - t0
        print(f"  ran {len(results)} cases in {dt:.2f}s")
        summarize(results, len(results))


if __name__ == "__main__":
    main()
