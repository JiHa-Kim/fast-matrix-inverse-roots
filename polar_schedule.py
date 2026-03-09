#!/usr/bin/env python3
# polar_optimal_schedule_scalar.py
#
# Exact scalar-schedule solver for the current project:
#   - no eigvalsh in the algorithm loop
#   - exact optimal schedule over the scalar model
#   - candidate family: DWH and safe ZOLO(r)
#   - objective: minimize outer steps, then small-side solve count, then final predicted kappa(O)
#   - runtime uses only:
#       * chunked fp64 Gram
#       * fp64 small-side Cholesky
#       * one tall right-update per outer step
#       * cheap trace/logdet certification
#
# Exact validation with eigvalsh is optional and metrics-only.

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch

try:
    import mpmath as mp
except Exception:
    mp = None

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


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype name: {name}")


def safe_exp(x: float) -> float:
    if x >= 709.0:
        return float("inf")
    return float(math.exp(x))


def acosh_exp(logu: float) -> float:
    if logu <= 0.0:
        return 0.0
    if logu < 20.0:
        u = math.exp(logu)
        return float(math.acosh(max(u, 1.0)))
    return float(logu + math.log(2.0))


def bf16_target(mode: str) -> float:
    u = 2.0**-8
    if mode == "aggressive":
        return float(1.0 + u)
    if mode == "robust":
        return float((1.0 + u) / (1.0 - u))
    raise ValueError(mode)


# ----------------------------- accurate small-side ops ----------------------


@torch.no_grad()
def gram_xtx_chunked_fp64(X: Tensor, chunk_rows: int) -> Tensor:
    m, n = X.shape
    S = torch.zeros((n, n), device=X.device, dtype=torch.float64)
    for i in range(0, m, chunk_rows):
        Xi = X[i : i + chunk_rows].float().to(torch.float64)
        S.addmm_(Xi.T, Xi)
    return symmetrize(S)


@torch.no_grad()
def chol_with_jitter_fp64(
    A: Tensor, jitter_rel: float, max_tries: int = 8
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
def cert_bound_trace_logdet(S: Tensor, jitter_rel: float) -> Tuple[float, float]:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    L, shift = chol_with_jitter_fp64(S, jitter_rel=jitter_rel)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()

    a = max(float((torch.trace(S) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    return float(safe_exp(eta_ub)), float(shift)


@torch.no_grad()
def exact_eigvalsh(S: Tensor, eig_device: str = "auto") -> Tensor:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    if eig_device == "cpu":
        use_cpu = True
    elif eig_device == "cuda":
        use_cpu = False
    else:
        use_cpu = (S.device.type != "cuda") or (n >= 4096)

    if use_cpu:
        evals = torch.linalg.eigvalsh(S.cpu())
        return evals.to(device=S.device)
    return torch.linalg.eigvalsh(S)


# ----------------------------- DWH scalar theory ----------------------------


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


def dwh_ell_next(ell: float) -> float:
    a, b, c = dwh_coeffs_from_ell(ell)
    return float(ell * (a + b * ell * ell) / (1.0 + c * ell * ell))


@torch.no_grad()
def dwh_small_right_update(
    S: Tensor, ell: float, jitter_rel: float
) -> Tuple[Tensor, float]:
    a, b, c = dwh_coeffs_from_ell(ell)

    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    M = symmetrize(I + float(c) * S)
    L, shift = chol_with_jitter_fp64(M, jitter_rel=jitter_rel)
    InvM = torch.cholesky_solve(I, L)

    alpha = float(b / c)
    beta = float(a - b / c)
    U = symmetrize(alpha * I + beta * InvM)
    return U, float(shift)


# ----------------------------- Zolo helpers ---------------------------------


@dataclasses.dataclass(frozen=True)
class ZoloCoeffs:
    r: int
    ell: float
    c_odd: Tuple[float, ...]
    c_even: Tuple[float, ...]
    a: Tuple[float, ...]
    mhat: float


@functools.lru_cache(maxsize=512)
def _zolo_coeffs_cached(r: int, ell_key: float, dps: int) -> ZoloCoeffs:
    if mp is None:
        raise RuntimeError("mpmath is required for Zolo coefficients")

    mp.mp.dps = int(dps)
    ell = mp.mpf(ell_key)

    kp = mp.sqrt(1 - ell * ell)
    m = kp * kp
    Kp = mp.ellipk(m)

    c_all = []
    for i in range(1, 2 * r + 1):
        u = mp.mpf(i) * Kp / mp.mpf(2 * r + 1)
        sn = mp.ellipfun("sn", u, m)
        cn = mp.ellipfun("cn", u, m)
        ci = ell * ell * (sn / cn) ** 2
        c_all.append(ci)

    c_odd = [c_all[2 * j] for j in range(r)]
    c_even = [c_all[2 * j + 1] for j in range(r)]

    mhat = mp.mpf(1)
    for j in range(r):
        mhat *= (1 + c_odd[j]) / (1 + c_even[j])

    a = []
    for j in range(r):
        x = c_odd[j]
        num = mp.mpf(1)
        den = mp.mpf(1)
        for k in range(r):
            num *= x - c_even[k]
            if k != j:
                den *= x - c_odd[k]
        a.append(-num / den)

    return ZoloCoeffs(
        r=int(r),
        ell=float(ell_key),
        c_odd=tuple(float(v) for v in c_odd),
        c_even=tuple(float(v) for v in c_even),
        a=tuple(float(v) for v in a),
        mhat=float(mhat),
    )


def zolo_coeffs_from_ell(r: int, ell: float, dps: int = 100) -> ZoloCoeffs:
    ell = float(min(max(ell, 1e-18), 1.0 - 1e-18))
    ell_key = float(f"{ell:.18e}")
    return _zolo_coeffs_cached(int(r), ell_key, int(dps))


def zolo_coeffs_from_pairs(
    r: int,
    ell: float,
    c_odd: Sequence[float],
    c_even: Sequence[float],
) -> ZoloCoeffs:
    c_odd_t = tuple(float(v) for v in c_odd)
    c_even_t = tuple(float(v) for v in c_even)
    if len(c_odd_t) != int(r) or len(c_even_t) != int(r):
        raise ValueError("pair count must match r")
    if not all(math.isfinite(v) and v > 0.0 for v in c_odd_t):
        raise ValueError("all c_odd values must be positive finite")
    if not all(math.isfinite(v) and v > 0.0 for v in c_even_t):
        raise ValueError("all c_even values must be positive finite")

    mhat = 1.0
    for co, ce in zip(c_odd_t, c_even_t):
        mhat *= (1.0 + co) / (1.0 + ce)

    a_vals = []
    for j, x in enumerate(c_odd_t):
        num = 1.0
        den = 1.0
        for k in range(int(r)):
            num *= x - c_even_t[k]
            if k != j:
                den *= x - c_odd_t[k]
        a_vals.append(-num / den)

    return ZoloCoeffs(
        r=int(r),
        ell=float(ell),
        c_odd=c_odd_t,
        c_even=c_even_t,
        a=tuple(float(v) for v in a_vals),
        mhat=float(mhat),
    )


def temper_zolo_coeffs_by_floor(coeffs: ZoloCoeffs, pole_floor: float) -> ZoloCoeffs:
    pole_floor = float(max(pole_floor, 0.0))
    if coeffs.r <= 0:
        return coeffs

    min_odd = min(float(v) for v in coeffs.c_odd)
    tau = max(0.0, pole_floor - min_odd)
    if tau == 0.0:
        return coeffs

    return zolo_coeffs_from_pairs(
        coeffs.r,
        coeffs.ell,
        c_odd=[float(v) + tau for v in coeffs.c_odd],
        c_even=[float(v) + tau for v in coeffs.c_even],
    )


def zolo_scalar_value(sigma: float, coeffs: ZoloCoeffs) -> float:
    x = float(sigma)
    x2 = x * x
    val = float(coeffs.mhat) * x
    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
        val *= (x2 + ce) / (x2 + co)
    return float(val)


def zolo_ell_next(ell: float, coeffs: ZoloCoeffs) -> float:
    return float(max(min(zolo_scalar_value(ell, coeffs), 1.0), 1e-300))


def zolo_safe_for_cholesky(
    ell: float, coeffs: ZoloCoeffs, shift_cond_max: float, max_a: float
) -> bool:
    if not all(math.isfinite(v) and v > 0.0 for v in coeffs.c_odd):
        return False
    if not all(math.isfinite(v) for v in coeffs.a):
        return False
    if max(abs(v) for v in coeffs.a) > float(max_a):
        return False

    ell2 = float(ell) * float(ell)
    max_cond = max((1.0 + c) / (ell2 + c) for c in coeffs.c_odd)
    if max_cond > float(shift_cond_max):
        return False

    return True


@torch.no_grad()
def zolo_small_right_update(
    S: Tensor, coeffs: ZoloCoeffs, jitter_rel: float
) -> Tuple[Tensor, float]:
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)

    U = I.clone()
    max_shift = 0.0
    for aj, codd in zip(coeffs.a, coeffs.c_odd):
        Z = symmetrize(S + float(codd) * I)
        L, shift = chol_with_jitter_fp64(Z, jitter_rel=jitter_rel)
        InvZ = torch.cholesky_solve(I, L)
        U = U + float(aj) * InvZ
        max_shift = max(max_shift, float(shift))

    U = symmetrize(float(coeffs.mhat) * U)
    return U, float(max_shift)


@torch.no_grad()
def zolo_product_step_chunked(
    X: Tensor,
    S: Tensor,
    coeffs: ZoloCoeffs,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    """
    Apply the Zolotarev step in product form:

      U = mhat * Π_j (S + c_even_j I) (S + c_odd_j I)^(-1)
        = mhat * Π_j [I + (c_even_j - c_odd_j) (S + c_odd_j I)^(-1)]

    This avoids the catastrophic cancellation of the partial-fraction sum
    realization while still using only Cholesky factorizations of shifted SPD
    systems.
    """

    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    X_work = X.to(dtype=out_dtype)
    max_shift = 0.0

    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
        Z = symmetrize(S + float(co) * I)
        L, shift = chol_with_jitter_fp64(Z, jitter_rel=jitter_rel)
        max_shift = max(max_shift, float(shift))
        delta = float(ce - co)

        X_next = torch.empty_like(X_work)
        for i in range(0, X_work.shape[0], rhs_chunk_rows):
            Xi = X_work[i : i + rhs_chunk_rows].float().to(torch.float64)
            Yi_t = torch.cholesky_solve(Xi.T.contiguous(), L)
            Yi = Yi_t.T
            Zi = Xi + delta * Yi
            X_next[i : i + rhs_chunk_rows] = Zi.to(dtype=out_dtype)
        X_work = X_next

    if out_dtype == torch.float64:
        X_work = float(coeffs.mhat) * X_work
    else:
        X_work = (float(coeffs.mhat) * X_work.float()).to(dtype=out_dtype)

    return X_work, float(max_shift)


# ----------------------------- exact schedule solver ------------------------


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str
    ell_in: float
    ell_out: float
    pred_kappa_after: float
    r: int = 0
    pole_floor: float = 0.0


@dataclasses.dataclass
class ScheduleInfo:
    objective_name: str
    cost_key: Tuple[float, ...]
    steps: List[StepSpec]


def solve_optimal_schedule_exact(
    ell0: float,
    target_kappa_O: float,
    max_steps: int,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
) -> ScheduleInfo:
    """
    Exact solver over the scalar candidate family.

    Objective is lexicographic:
      1) minimize outer steps
      2) minimize total small-side solve count (DWH=1, Zolo=r)
      3) minimize final predicted kappa(O)
    """
    memo = {}

    def key_ell(x: float) -> float:
        return float(f"{x:.16e}")

    def rec(ell: float, steps_left: int):
        ell = key_ell(ell)
        if 1.0 / max(ell, 1e-300) <= float(target_kappa_O):
            return (0, 0, 1.0 / max(ell, 1e-300)), []

        if steps_left == 0:
            return (10**9, 10**9, float("inf")), []

        key = (ell, steps_left)
        if key in memo:
            return memo[key]

        best_cost = (10**9, 10**9, float("inf"))
        best_sched: List[StepSpec] = []

        # DWH candidate
        ell_dwh = dwh_ell_next(ell)
        step = StepSpec(
            kind="DWH",
            ell_in=float(ell),
            ell_out=float(ell_dwh),
            pred_kappa_after=float(1.0 / max(ell_dwh, 1e-300)),
            r=1,
        )
        rem_cost, rem_sched = rec(ell_dwh, steps_left - 1)
        cand_cost = (1 + rem_cost[0], 1 + rem_cost[1], rem_cost[2])
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_sched = [step] + rem_sched

        # Zolo candidates
        for r in sorted(int(v) for v in zolo_r_values):
            coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(zolo_coeff_dps))
            if not zolo_safe_for_cholesky(
                float(ell), coeffs, zolo_shift_cond_max, zolo_max_a
            ):
                continue

            ell_z = zolo_ell_next(float(ell), coeffs)
            step = StepSpec(
                kind="ZOLO",
                ell_in=float(ell),
                ell_out=float(ell_z),
                pred_kappa_after=float(1.0 / max(ell_z, 1e-300)),
                r=int(r),
            )
            rem_cost, rem_sched = rec(ell_z, steps_left - 1)
            cand_cost = (1 + rem_cost[0], int(r) + rem_cost[1], rem_cost[2])
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_sched = [step] + rem_sched

        memo[key] = (best_cost, best_sched)
        return memo[key]

    cost, sched = rec(float(ell0), int(max_steps))
    return ScheduleInfo(
        objective_name="exact_optimal",
        cost_key=cost,
        steps=sched,
    )


def solve_two_step_exact_cholesky(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    tighten_fraction: float,
) -> ScheduleInfo:
    """
    Search exact two-step Zolotarev schedules meant for the product-form
    Cholesky realization. Since the product form is much more stable than the
    partial-fraction sum, the best schedule is usually just the smallest orders
    that hit a slightly tightened target.
    """

    ell0 = float(ell0)
    target_kappa_O = float(target_kappa_O)
    tighten_fraction = float(min(max(tighten_fraction, 0.0), 1.0))
    effective_target = float(
        1.0 + (target_kappa_O - 1.0) * (1.0 - tighten_fraction)
    )

    if 1.0 / max(ell0, 1e-300) <= effective_target:
        return ScheduleInfo(
            objective_name="two_step_exact_cholesky",
            cost_key=(0.0, 1.0 / max(ell0, 1e-300), 0.0),
            steps=[],
        )

    best_key = None
    best_sched = None

    orders = sorted(int(v) for v in zolo_r_values)
    for r1 in orders:
        coeffs1 = zolo_coeffs_from_ell(r1, ell0, dps=int(zolo_coeff_dps))
        ell1 = zolo_ell_next(ell0, coeffs1)
        step1 = StepSpec(
            kind="ZOLO",
            ell_in=ell0,
            ell_out=float(ell1),
            pred_kappa_after=float(1.0 / max(ell1, 1e-300)),
            r=int(r1),
        )

        if step1.pred_kappa_after <= effective_target:
            key = (float(r1), float(step1.pred_kappa_after), 0.0)
            if best_key is None or key < best_key:
                best_key = key
                best_sched = [step1]
            continue

        for r2 in orders:
            coeffs2 = zolo_coeffs_from_ell(r2, ell1, dps=int(zolo_coeff_dps))
            ell2 = zolo_ell_next(ell1, coeffs2)
            step2 = StepSpec(
                kind="ZOLO",
                ell_in=float(ell1),
                ell_out=float(ell2),
                pred_kappa_after=float(1.0 / max(ell2, 1e-300)),
                r=int(r2),
            )
            if step2.pred_kappa_after > effective_target:
                continue

            key = (float(r1 + r2), float(step2.pred_kappa_after), float(r2))
            if best_key is None or key < best_key:
                best_key = key
                best_sched = [step1, step2]

    if best_key is None or best_sched is None:
        return ScheduleInfo(
            objective_name="two_step_exact_cholesky",
            cost_key=(float("inf"), float("inf"), float("inf")),
            steps=[],
        )

    return ScheduleInfo(
        objective_name="two_step_exact_cholesky",
        cost_key=best_key,
        steps=best_sched,
    )


def dwh_stability_metrics(ell: float) -> Tuple[float, float]:
    a, b, c = dwh_coeffs_from_ell(ell)
    ell2 = float(ell) * float(ell)
    solve_cond = float((1.0 + c) / (ell2 + c))
    combine_scale = float(max(abs(b / c), abs(a - b / c), 1.0))
    return solve_cond, combine_scale


def zolo_stability_metrics(ell: float, coeffs: ZoloCoeffs) -> Tuple[float, float]:
    ell2 = float(ell) * float(ell)
    solve_cond = float(max((1.0 + c) / (ell2 + c) for c in coeffs.c_odd))
    combine_scale = float(max(max(abs(v) for v in coeffs.a), abs(coeffs.mhat), 1.0))
    return solve_cond, combine_scale


def zolo_badness_metrics(ell: float, coeffs: ZoloCoeffs) -> Tuple[float, float, float]:
    ell2 = float(ell) * float(ell)
    max_cond = float(max((1.0 + c) / (ell2 + c) for c in coeffs.c_odd))
    max_a = float(max(abs(v) for v in coeffs.a))
    cancel = float(sum(abs(v) for v in coeffs.a))
    return max_cond, max_a, cancel


def tempered_zolo_candidate(
    ell: float,
    r: int,
    pole_floor: float,
    dps: int,
) -> Tuple[StepSpec, ZoloCoeffs, Tuple[float, float, float]]:
    coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(dps))
    coeffs = temper_zolo_coeffs_by_floor(coeffs, pole_floor=float(pole_floor))
    ell_out = zolo_ell_next(float(ell), coeffs)
    max_cond, max_a, cancel = zolo_badness_metrics(float(ell), coeffs)
    step = StepSpec(
        kind="TZOLO",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=int(r),
        pole_floor=float(max(pole_floor, 0.0)),
    )
    return step, coeffs, (max_cond, max_a, cancel)


def step_small_solve_count(step: StepSpec) -> int:
    return 1 if step.kind == "DWH" else int(step.r)


def enumerate_safe_scalar_steps(
    ell: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
) -> List[Tuple[StepSpec, float, float]]:
    out: List[Tuple[StepSpec, float, float]] = []

    ell_dwh = dwh_ell_next(ell)
    dwh_cond, dwh_scale = dwh_stability_metrics(ell)
    out.append(
        (
            StepSpec(
                kind="DWH",
                ell_in=float(ell),
                ell_out=float(ell_dwh),
                pred_kappa_after=float(1.0 / max(ell_dwh, 1e-300)),
                r=1,
            ),
            dwh_cond,
            dwh_scale,
        )
    )

    for r in sorted(int(v) for v in zolo_r_values):
        coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(zolo_coeff_dps))
        if not zolo_safe_for_cholesky(
            float(ell), coeffs, zolo_shift_cond_max, zolo_max_a
        ):
            continue

        ell_z = zolo_ell_next(float(ell), coeffs)
        z_cond, z_scale = zolo_stability_metrics(float(ell), coeffs)
        out.append(
            (
                StepSpec(
                    kind="ZOLO",
                    ell_in=float(ell),
                    ell_out=float(ell_z),
                    pred_kappa_after=float(1.0 / max(ell_z, 1e-300)),
                    r=int(r),
                ),
                z_cond,
                z_scale,
            )
        )

    return out


def solve_two_step_schedule_sufficient(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
    switch_ell_min: float,
) -> ScheduleInfo:
    """
    Search a numerically conservative two-step schedule.

    The objective is not minimax optimality. Instead, among schedules that hit the
    target in at most two outer steps, prefer the one with the mildest small-side
    solve conditioning, then the mildest coefficient/combination scale, then the
    lowest small-side solve count, then the best final predicted kappa(O).

    `switch_ell_min` expresses the key heuristic: once the first step raises ell
    into a moderate regime, a second-step Zolo can be both aggressive and stable.
    """

    if 1.0 / max(float(ell0), 1e-300) <= float(target_kappa_O):
        return ScheduleInfo(
            objective_name="two_step_sufficient",
            cost_key=(1.0, 1.0, 0.0, 1.0 / max(float(ell0), 1e-300)),
            steps=[],
        )

    first_steps = enumerate_safe_scalar_steps(
        ell=float(ell0),
        zolo_r_values=zolo_r_values,
        zolo_coeff_dps=zolo_coeff_dps,
        zolo_shift_cond_max=zolo_shift_cond_max,
        zolo_max_a=zolo_max_a,
    )

    def pick(require_switch: bool) -> Tuple[Tuple[float, ...], List[StepSpec]] | None:
        best_score = None
        best_sched = None

        for step1, cond1, scale1 in first_steps:
            if require_switch and step1.ell_out < float(switch_ell_min):
                continue

            if step1.pred_kappa_after <= float(target_kappa_O):
                score = (
                    float(cond1),
                    float(scale1),
                    float(step_small_solve_count(step1)),
                    float(step1.pred_kappa_after),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_sched = [step1]
                continue

            second_steps = enumerate_safe_scalar_steps(
                ell=float(step1.ell_out),
                zolo_r_values=zolo_r_values,
                zolo_coeff_dps=zolo_coeff_dps,
                zolo_shift_cond_max=zolo_shift_cond_max,
                zolo_max_a=zolo_max_a,
            )
            for step2, cond2, scale2 in second_steps:
                if step2.pred_kappa_after > float(target_kappa_O):
                    continue

                score = (
                    float(max(cond1, cond2)),
                    float(max(scale1, scale2)),
                    float(step_small_solve_count(step1) + step_small_solve_count(step2)),
                    float(step2.pred_kappa_after),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_sched = [step1, step2]

        if best_score is None or best_sched is None:
            return None
        return best_score, best_sched

    picked = pick(require_switch=True)
    if picked is None:
        picked = pick(require_switch=False)
    if picked is None:
        return ScheduleInfo(
            objective_name="two_step_sufficient",
            cost_key=(float("inf"), float("inf"), float("inf"), float("inf")),
            steps=[],
        )

    cost, sched = picked
    return ScheduleInfo(
        objective_name="two_step_sufficient",
        cost_key=cost,
        steps=sched,
    )


def solve_two_step_tempered_cholesky(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    pole_floor_values: Sequence[float],
    max_cond_allowed: float,
    max_a_allowed: float,
    max_cancel_allowed: float,
    tighten_fraction: float,
    require_monotone_gain: bool = True,
) -> ScheduleInfo:
    """
    Search two tempered Zolo steps executed only through Cholesky solves.

    Each step starts from the exact Zolotarev poles/zeros for the current ell and
    then inflates both pole and zero sets by the same additive shift so that the
    smallest pole reaches `pole_floor`. This trades away minimax optimality for a
    milder partial-fraction representation that is cheaper to realize stably.
    """

    ell0 = float(ell0)
    target_kappa_O = float(target_kappa_O)
    tighten_fraction = float(min(max(tighten_fraction, 0.0), 1.0))
    effective_target = float(
        1.0 + (target_kappa_O - 1.0) * (1.0 - tighten_fraction)
    )

    if 1.0 / max(ell0, 1e-300) <= effective_target:
        return ScheduleInfo(
            objective_name="two_step_tempered_cholesky",
            cost_key=(0.0, 0.0, 0.0, 0.0, 1.0 / max(ell0, 1e-300)),
            steps=[],
        )

    best_key = None
    best_sched = None

    floors = sorted(float(max(v, 0.0)) for v in pole_floor_values)
    orders = sorted(int(v) for v in zolo_r_values)

    for r1 in orders:
        for floor1 in floors:
            step1, _coeffs1, metrics1 = tempered_zolo_candidate(
                ell=ell0,
                r=r1,
                pole_floor=floor1,
                dps=zolo_coeff_dps,
            )
            cond1, a1, cancel1 = metrics1
            if cond1 > float(max_cond_allowed):
                continue
            if a1 > float(max_a_allowed):
                continue
            if cancel1 > float(max_cancel_allowed):
                continue
            if require_monotone_gain and step1.ell_out <= ell0:
                continue

            for r2 in orders:
                for floor2 in floors:
                    step2, _coeffs2, metrics2 = tempered_zolo_candidate(
                        ell=step1.ell_out,
                        r=r2,
                        pole_floor=floor2,
                        dps=zolo_coeff_dps,
                    )
                    cond2, a2, cancel2 = metrics2
                    if cond2 > float(max_cond_allowed):
                        continue
                    if a2 > float(max_a_allowed):
                        continue
                    if cancel2 > float(max_cancel_allowed):
                        continue
                    if require_monotone_gain and step2.ell_out <= step1.ell_out:
                        continue
                    if step2.pred_kappa_after > effective_target:
                        continue

                    key = (
                        float(r1 + r2),
                        float(max(cond1, cond2)),
                        float(max(a1, a2)),
                        float(max(cancel1, cancel2)),
                        float(step2.pred_kappa_after),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_sched = [step1, step2]

    if best_key is None or best_sched is None:
        return ScheduleInfo(
            objective_name="two_step_tempered_cholesky",
            cost_key=(float("inf"),) * 5,
            steps=[],
        )

    return ScheduleInfo(
        objective_name="two_step_tempered_cholesky",
        cost_key=best_key,
        steps=best_sched,
    )


# ----------------------------- tall update ----------------------------------


@torch.no_grad()
def apply_right_small_chunked(
    X: Tensor, U: Tensor, rhs_chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    m, n = X.shape
    X_next = torch.empty((m, n), device=X.device, dtype=out_dtype)
    U_work = U if out_dtype == torch.float64 else U.float()

    for i in range(0, m, rhs_chunk_rows):
        Xi = X[i : i + rhs_chunk_rows].to(dtype=U_work.dtype)
        Zi = Xi @ U_work
        X_next[i : i + rhs_chunk_rows] = Zi.to(dtype=out_dtype)

    return X_next


# ----------------------------- synthetic matrices ---------------------------


def make_matrix_from_singulars(
    m: int, singulars: Tensor, seed: int, device: str, storage_dtype: torch.dtype
) -> Tensor:
    n = int(singulars.numel())

    seed_all(seed)
    U, _ = torch.linalg.qr(
        torch.randn(m, n, device=device, dtype=torch.float32),
        mode="reduced",
    )

    seed_all(seed + 1)
    V, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=torch.float32),
        mode="reduced",
    )

    G = (U * singulars.to(device=device, dtype=torch.float32)) @ V.T
    return G.to(dtype=storage_dtype)


def make_spectrum_bank(
    n: int, kappa_G: float, bank_size: int, seed: int
) -> List[Tensor]:
    sig_max = 1.0
    sig_min = 1.0 / float(kappa_G)

    out: List[Tensor] = []
    out.append(
        torch.logspace(0.0, math.log10(sig_min), n, base=10.0, dtype=torch.float32)
    )

    t = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs1 = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (t**p)
        logs2 = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs1))
        out.append(torch.exp(logs2))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        s = torch.full((n,), sig_min, dtype=torch.float32)
        s[:r] = sig_max
        out.append(s)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(sig_min) * x for x in u], dtype=torch.float32)
        s = torch.exp(logs)
        s[0] = sig_max
        s[-1] = sig_min
        out.append(s)

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


# ----------------------------- run core -------------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_cert: float
    final_kO_exact: float
    steps: int
    dwh_steps: int
    zolo_steps: int
    guards: int
    fallbacks: int
    last_step_kind: str
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_cert: float
    ms_total: float


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    jitter_rel: float,
    cert_jitter_rel: float,
    tf32: bool,
    validate_exact: bool,
    exact_validate_threshold: int,
    exact_validate_device: str,
    zolo_coeff_dps: int,
    zolo_realization: str,
    stop_on_cert: bool,
) -> RunSummary:
    device = G_storage.device

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    X = G_storage.to(dtype=iter_dtype)

    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    ms_cert_sum = 0.0

    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"

    final_kO_cert = float("inf")
    final_kO_exact = float("nan")

    # Run the optimal schedule first.
    for step in schedule:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
        ms_gram_sum += ms_gram

        try:
            if step.kind == "DWH":
                ms_solve, (U, shift) = cuda_time_ms(
                    lambda: dwh_small_right_update(S, step.ell_in, jitter_rel)
                )
                dwh_steps += 1
                last_step_kind = "DWH"
            else:
                coeffs = zolo_coeffs_from_ell(step.r, step.ell_in, dps=zolo_coeff_dps)
                if step.kind == "TZOLO":
                    coeffs = temper_zolo_coeffs_by_floor(coeffs, step.pole_floor)
                if zolo_realization == "product":
                    ms_solve, (X, shift) = cuda_time_ms(
                        lambda: zolo_product_step_chunked(
                            X=X,
                            S=S,
                            coeffs=coeffs,
                            rhs_chunk_rows=rhs_chunk_rows,
                            jitter_rel=jitter_rel,
                            out_dtype=iter_dtype,
                        )
                    )
                else:
                    ms_solve, (U, shift) = cuda_time_ms(
                        lambda: zolo_small_right_update(S, coeffs, jitter_rel)
                    )
                zolo_steps += 1
                if step.kind == "TZOLO":
                    last_step_kind = f"TZOLO(r={step.r},floor={step.pole_floor:.3e})"
                else:
                    last_step_kind = f"ZOLO(r={step.r})"

            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)

        except Exception:
            # Fail closed to DWH at the same scalar ell_in.
            fallbacks += 1
            ms_solve, (U, shift) = cuda_time_ms(
                lambda: dwh_small_right_update(S, step.ell_in, jitter_rel)
            )
            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)
            dwh_steps += 1
            last_step_kind = "DWH(fallback)"

        if step.kind == "DWH" or zolo_realization == "partial":
            ms_upd, X = cuda_time_ms(
                lambda: apply_right_small_chunked(X, U, rhs_chunk_rows, iter_dtype)
            )
            ms_upd_sum += ms_upd

    # Certify after the chosen schedule.
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
    ms_gram_sum += ms_gram

    ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
        lambda: cert_bound_trace_logdet(S, cert_jitter_rel)
    )
    ms_cert_sum += ms_cert
    guards += int(cert_shift > 0.0)
    final_kO_cert = float(kO_cert)

    if validate_exact and S.shape[0] <= int(exact_validate_threshold):
        evals = exact_eigvalsh(S, eig_device=exact_validate_device)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        final_kO_exact = float(math.sqrt(lam_max / lam_min))

    steps_used = len(schedule)

    # Optional extra polishing if you want cert-driven continuation.
    if stop_on_cert:
        ell = schedule[-1].ell_out if schedule else 1.0
        while final_kO_cert > target_kappa_O and steps_used < 16:
            ms_solve, (U, shift) = cuda_time_ms(
                lambda: dwh_small_right_update(S, ell, jitter_rel)
            )
            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)
            dwh_steps += 1
            last_step_kind = "DWH(polish)"

            ms_upd, X = cuda_time_ms(
                lambda: apply_right_small_chunked(X, U, rhs_chunk_rows, iter_dtype)
            )
            ms_upd_sum += ms_upd

            ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
            ms_gram_sum += ms_gram

            ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
                lambda: cert_bound_trace_logdet(S, cert_jitter_rel)
            )
            ms_cert_sum += ms_cert
            guards += int(cert_shift > 0.0)
            final_kO_cert = float(kO_cert)

            ell = dwh_ell_next(ell)
            steps_used += 1

    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
    return RunSummary(
        success=(final_kO_cert <= target_kappa_O) if stop_on_cert else True,
        final_kO_cert=float(final_kO_cert),
        final_kO_exact=float(final_kO_exact),
        steps=steps_used,
        dwh_steps=dwh_steps,
        zolo_steps=zolo_steps,
        guards=guards,
        fallbacks=fallbacks,
        last_step_kind=last_step_kind,
        ms_gram=ms_gram_sum,
        ms_solve=ms_solve_sum,
        ms_upd=ms_upd_sum,
        ms_cert=ms_cert_sum,
        ms_total=ms_total,
    )


# ----------------------------- CLI ------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)

    ap.add_argument(
        "--target_mode", choices=["aggressive", "robust", "custom"], default="robust"
    )
    ap.add_argument("--target_kappa_O", type=float, default=0.0)

    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument(
        "--schedule_mode",
        choices=[
            "optimal",
            "two_step_exact_cholesky",
            "two_step_sufficient",
            "two_step_tempered_cholesky",
        ],
        default="two_step_exact_cholesky",
    )

    ap.add_argument(
        "--input_dtype", choices=["float32", "bfloat16", "float64"], default="float32"
    )
    ap.add_argument(
        "--iter_dtype", choices=["float32", "bfloat16", "float64"], default="float32"
    )

    ap.add_argument("--gram_chunk_rows", type=int, default=2048)
    ap.add_argument("--rhs_chunk_rows", type=int, default=2048)
    ap.add_argument("--jitter_rel", type=float, default=1e-15)
    ap.add_argument("--cert_jitter_rel", type=float, default=1e-15)
    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--ell0", type=float, default=0.0, help="If 0, uses 1 / kappa_G")
    ap.add_argument("--zolo_r_values", type=str, default="2,3,4,5,6,8,10,12")
    ap.add_argument("--zolo_coeff_dps", type=int, default=100)
    ap.add_argument(
        "--zolo_realization", choices=["product", "partial"], default="product"
    )
    ap.add_argument("--zolo_shift_cond_max", type=float, default=1e4)
    ap.add_argument("--zolo_max_a", type=float, default=1e6)
    ap.add_argument("--two_step_switch_ell_min", type=float, default=1e-2)
    ap.add_argument(
        "--tempered_pole_floors",
        type=str,
        default="0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
    )
    ap.add_argument("--tempered_max_cond", type=float, default=2e4)
    ap.add_argument("--tempered_max_a", type=float, default=64.0)
    ap.add_argument("--tempered_max_cancel", type=float, default=256.0)
    ap.add_argument("--tempered_tighten_fraction", type=float, default=0.5)

    ap.add_argument("--validate_exact", action="store_true", default=False)
    ap.add_argument("--exact_validate_threshold", type=int, default=1024)
    ap.add_argument(
        "--exact_validate_device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    ap.add_argument("--stop_on_cert", action="store_true", default=False)

    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if mp is None:
        raise RuntimeError("This script requires mpmath for Zolo coefficients")

    input_dtype = dtype_from_name(args.input_dtype)
    iter_dtype = dtype_from_name(args.iter_dtype)
    zolo_r_values = [int(x.strip()) for x in args.zolo_r_values.split(",") if x.strip()]
    tempered_pole_floors = [
        float(x.strip()) for x in args.tempered_pole_floors.split(",") if x.strip()
    ]

    ell0 = float(args.ell0) if args.ell0 > 0.0 else (1.0 / float(args.kappa_G))

    if args.target_mode == "custom":
        if args.target_kappa_O <= 0.0:
            raise ValueError(
                "--target_kappa_O must be positive when target_mode=custom"
            )
        target_kappa_O = float(args.target_kappa_O)
    else:
        target_kappa_O = bf16_target(args.target_mode)

    if args.schedule_mode == "optimal":
        sched_info = solve_optimal_schedule_exact(
            ell0=ell0,
            target_kappa_O=target_kappa_O,
            max_steps=args.max_steps,
            zolo_r_values=zolo_r_values,
            zolo_coeff_dps=args.zolo_coeff_dps,
            zolo_shift_cond_max=args.zolo_shift_cond_max,
            zolo_max_a=args.zolo_max_a,
        )
    elif args.schedule_mode == "two_step_exact_cholesky":
        sched_info = solve_two_step_exact_cholesky(
            ell0=ell0,
            target_kappa_O=target_kappa_O,
            zolo_r_values=zolo_r_values,
            zolo_coeff_dps=args.zolo_coeff_dps,
            tighten_fraction=args.tempered_tighten_fraction,
        )
    elif args.schedule_mode == "two_step_sufficient":
        sched_info = solve_two_step_schedule_sufficient(
            ell0=ell0,
            target_kappa_O=target_kappa_O,
            zolo_r_values=zolo_r_values,
            zolo_coeff_dps=args.zolo_coeff_dps,
            zolo_shift_cond_max=args.zolo_shift_cond_max,
            zolo_max_a=args.zolo_max_a,
            switch_ell_min=args.two_step_switch_ell_min,
        )
    else:
        sched_info = solve_two_step_tempered_cholesky(
            ell0=ell0,
            target_kappa_O=target_kappa_O,
            zolo_r_values=zolo_r_values,
            zolo_coeff_dps=args.zolo_coeff_dps,
            pole_floor_values=tempered_pole_floors,
            max_cond_allowed=args.tempered_max_cond,
            max_a_allowed=args.tempered_max_a,
            max_cancel_allowed=args.tempered_max_cancel,
            tighten_fraction=args.tempered_tighten_fraction,
        )
    schedule = sched_info.steps
    if (
        args.schedule_mode in ["two_step_exact_cholesky", "two_step_tempered_cholesky"]
        and not schedule
    ):
        raise RuntimeError(
            "no two-step Cholesky schedule found under the current constraints"
        )

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  "
        f"target_mode={args.target_mode}  target_kappa(O)<={target_kappa_O:.8g}"
    )
    print(
        "knobs: "
        f"max_steps={args.max_steps} input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"gram_chunk_rows={args.gram_chunk_rows} rhs_chunk_rows={args.rhs_chunk_rows} "
        f"jitter_rel={args.jitter_rel:g} cert_jitter_rel={args.cert_jitter_rel:g} tf32={args.tf32}"
    )
    print(
        "control: "
        f"ell0={ell0:.6g} zolo_r_values={zolo_r_values} zolo_coeff_dps={args.zolo_coeff_dps} "
        f"zolo_realization={args.zolo_realization} "
        f"zolo_shift_cond_max={args.zolo_shift_cond_max:g} zolo_max_a={args.zolo_max_a:g} "
        f"two_step_switch_ell_min={args.two_step_switch_ell_min:g} "
        f"tempered_pole_floors={tempered_pole_floors} "
        f"tempered_max_cond={args.tempered_max_cond:g} "
        f"tempered_max_a={args.tempered_max_a:g} "
        f"tempered_max_cancel={args.tempered_max_cancel:g} "
        f"tempered_tighten_fraction={args.tempered_tighten_fraction:g} "
        f"stop_on_cert={args.stop_on_cert}"
    )
    if sched_info.objective_name == "exact_optimal":
        print(
            "optimal schedule objective: minimize outer steps, then small-side solve count, "
            "then final predicted kappa(O)"
        )
    elif sched_info.objective_name == "two_step_exact_cholesky":
        print(
            "two-step exact-Cholesky objective: hit a tightened target using exact "
            "Zolo maps with the product-form Cholesky realization, minimizing total "
            "solve count first"
        )
    elif sched_info.objective_name == "two_step_sufficient":
        print(
            "two-step sufficient objective: hit target in <=2 steps while minimizing "
            "worst small-side conditioning, then combination scale, then solve count"
        )
    else:
        print(
            "two-step tempered-Cholesky objective: hit a tightened target using only "
            "tempered Zolo steps while minimizing total solve count, then worst-step "
            "conditioning, weight size, and cancellation"
        )
    print("chosen schedule:")
    for i, st in enumerate(schedule, 1):
        if st.kind == "DWH":
            print(
                f"  step {i}: DWH        ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}"
            )
        elif st.kind == "TZOLO":
            print(
                f"  step {i}: TZOLO(r={st.r}, floor={st.pole_floor:.3e}) "
                f"ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}"
            )
        else:
            print(
                f"  step {i}: ZOLO(r={st.r}) ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}"
            )
    if sched_info.objective_name == "exact_optimal":
        print(
            f"schedule cost key: steps={int(sched_info.cost_key[0])} "
            f"small_cost={int(sched_info.cost_key[1])} "
            f"final_pred={sched_info.cost_key[2]:.8g}"
        )
    elif sched_info.objective_name == "two_step_exact_cholesky":
        print(
            f"schedule cost key: small_cost={sched_info.cost_key[0]:.8g} "
            f"final_pred={sched_info.cost_key[1]:.8g} "
            f"tie_break_r2={sched_info.cost_key[2]:.8g}"
        )
    elif sched_info.objective_name == "two_step_sufficient":
        print(
            f"schedule cost key: worst_cond={sched_info.cost_key[0]:.8g} "
            f"worst_scale={sched_info.cost_key[1]:.8g} "
            f"small_cost={sched_info.cost_key[2]:.8g} "
            f"final_pred={sched_info.cost_key[3]:.8g}"
        )
    else:
        print(
            f"schedule cost key: small_cost={sched_info.cost_key[0]:.8g} "
            f"worst_cond={sched_info.cost_key[1]:.8g} "
            f"worst_max_a={sched_info.cost_key[2]:.8g} "
            f"worst_cancel={sched_info.cost_key[3]:.8g} "
            f"final_pred={sched_info.cost_key[4]:.8g}"
        )

    def make_case(m: int, n: int, case_seed: int) -> Tensor:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=case_seed + n)
        return make_matrix_from_singulars(
            m=m,
            singulars=spectra[0],
            seed=case_seed,
            device=args.device,
            storage_dtype=input_dtype,
        )

    def run_case(G: Tensor) -> RunSummary:
        return run_one_case(
            G_storage=G,
            target_kappa_O=target_kappa_O,
            schedule=schedule,
            iter_dtype=iter_dtype,
            gram_chunk_rows=args.gram_chunk_rows,
            rhs_chunk_rows=args.rhs_chunk_rows,
            jitter_rel=args.jitter_rel,
            cert_jitter_rel=args.cert_jitter_rel,
            tf32=args.tf32,
            validate_exact=args.validate_exact,
            exact_validate_threshold=args.exact_validate_threshold,
            exact_validate_device=args.exact_validate_device,
            zolo_coeff_dps=args.zolo_coeff_dps,
            zolo_realization=args.zolo_realization,
            stop_on_cert=args.stop_on_cert,
        )

    if args.mode == "demo":
        G = make_case(args.m, args.n, args.seed)
        res = run_case(G)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} "
            f"final_kappa(O)_cert={res.final_kO_cert:.8g} exact={res.final_kO_exact:.8g} "
            f"steps={res.steps} dwh_steps={res.dwh_steps} zolo_steps={res.zolo_steps} "
            f"guards={res.guards} fallbacks={res.fallbacks} last_step={res.last_step_kind}"
        )
        print(
            f"  ms total={res.ms_total:.3f} "
            f"(gram={res.ms_gram:.3f} solve={res.ms_solve:.3f} upd={res.ms_upd:.3f} cert={res.ms_cert:.3f})"
        )
        return

    if args.mode == "bank":
        finals = []
        finals_exact = []
        steps = []
        dwh_steps = []
        zolo_steps = []
        guards = []
        fallbacks = []
        ms_total = []

        for i in range(args.bank_size):
            try:
                G = make_case(args.m, args.n, args.seed + 1000 + i)
                res = run_case(G)
                finals.append(res.final_kO_cert)
                finals_exact.append(res.final_kO_exact)
                steps.append(res.steps)
                dwh_steps.append(res.dwh_steps)
                zolo_steps.append(res.zolo_steps)
                guards.append(res.guards)
                fallbacks.append(res.fallbacks)
                ms_total.append(res.ms_total)
                del G
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                steps.append(0)
                dwh_steps.append(0)
                zolo_steps.append(0)
                guards.append(0)
                fallbacks.append(0)
                ms_total.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target: {sum(1 for x in finals if x <= target_kappa_O)}/{len(finals)}"
        )
        print(
            f"  worst kappa(O)_cert: {max(finals):.8g}  median: {pct(finals, 0.5):.8g}  p90: {pct(finals, 0.9):.8g}"
        )
        if any(math.isfinite(x) for x in finals_exact):
            print(
                f"  exact kappa(O) median: {pct(finals_exact, 0.5):.8g}  p90: {pct(finals_exact, 0.9):.8g}"
            )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(
            f"  dwh_steps median: {pct(dwh_steps, 0.5):.6g}  p90: {pct(dwh_steps, 0.9):.6g}"
        )
        print(
            f"  zolo_steps median: {pct(zolo_steps, 0.5):.6g}  p90: {pct(zolo_steps, 0.9):.6g}"
        )
        print(
            f"  fallbacks median: {pct(fallbacks, 0.5):.6g}  p90: {pct(fallbacks, 0.9):.6g}"
        )
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
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

        finals = []
        finals_exact = []
        steps_used = []
        dwh_steps_used = []
        zolo_steps_used = []
        guards_used = []
        fallbacks_used = []
        ms_total = []
        ms_gram = []
        ms_solve = []
        ms_upd = []
        ms_cert = []
        successes = 0

        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G = make_case(m, n, args.seed + 10000 + i)
                res = run_case(G)
                finals.append(res.final_kO_cert)
                finals_exact.append(res.final_kO_exact)
                steps_used.append(res.steps)
                dwh_steps_used.append(res.dwh_steps)
                zolo_steps_used.append(res.zolo_steps)
                guards_used.append(res.guards)
                fallbacks_used.append(res.fallbacks)
                successes += int(res.final_kO_cert <= target_kappa_O)
                ms_total.append(res.ms_total)
                ms_gram.append(res.ms_gram)
                ms_solve.append(res.ms_solve)
                ms_upd.append(res.ms_upd)
                ms_cert.append(res.ms_cert)
                del G
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                steps_used.append(0)
                dwh_steps_used.append(0)
                zolo_steps_used.append(0)
                guards_used.append(0)
                fallbacks_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_cert.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                steps_used.append(0)
                dwh_steps_used.append(0)
                zolo_steps_used.append(0)
                guards_used.append(0)
                fallbacks_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_cert.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  ran {args.suite_cases} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{args.suite_cases}")
        print(
            f"  worst kappa(O)_cert: {max(finals):.8g}  median: {pct(finals, 0.5):.8g}  p90: {pct(finals, 0.9):.8g}"
        )
        if any(math.isfinite(x) for x in finals_exact):
            print(
                f"  exact kappa(O) median: {pct(finals_exact, 0.5):.8g}  p90: {pct(finals_exact, 0.9):.8g}"
            )
        print(
            f"  steps median: {pct(steps_used, 0.5):.6g}  p90: {pct(steps_used, 0.9):.6g}"
        )
        print(
            f"  dwh_steps median: {pct(dwh_steps_used, 0.5):.6g}  p90: {pct(dwh_steps_used, 0.9):.6g}"
        )
        print(
            f"  zolo_steps median: {pct(zolo_steps_used, 0.5):.6g}  p90: {pct(zolo_steps_used, 0.9):.6g}"
        )
        print(
            f"  fallbacks median: {pct(fallbacks_used, 0.5):.6g}  p90: {pct(fallbacks_used, 0.9):.6g}"
        )
        print(
            f"  guards median: {pct(guards_used, 0.5):.6g}  p90: {pct(guards_used, 0.9):.6g}"
        )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms gram  median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}"
        )
        print(
            f"    ms solve median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}"
        )
        print(
            f"    ms upd   median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}"
        )
        print(
            f"    ms cert  median: {pct(ms_cert, 0.5):.3f}  p90: {pct(ms_cert, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
