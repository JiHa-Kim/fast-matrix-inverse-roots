#!/usr/bin/env python3
"""
Principled bf16-safe polynomial design for Phase 1 (x in [ell, 1]):

- Choose degree d Chebyshev series q(x) on [ell,1].
- Optimize coefficients via LP to maximize min g(x)=sqrt(x) q(x),
  subject to upper safety g(x) <= 1 - mu on a proxy set (dense real grid + bf16 values).
- Verify safety exhaustively on ALL bf16 values in [ell,1] using a bf16-rounded Chebyshev recurrence evaluator.
- Binary search the smallest mu that passes bf16 verification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import linprog


# -------------------------
# bf16 rounding (float32 <-> bf16 <-> float32), round-to-nearest-even
# -------------------------


def bf16_round_f32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    u = x.view(np.uint32)

    # Keep NaN/Inf unchanged
    is_special = (u & 0x7F800000) == 0x7F800000

    lsb = (u >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb.astype(np.uint32)
    ur = (u + bias) & np.uint32(0xFFFF0000)

    out = ur.view(np.float32)
    out[is_special] = x[is_special]
    return out


def all_bf16_values_in_interval(a: float, b: float) -> np.ndarray:
    """
    Enumerate all finite bf16 representables in [a,b], returned as float32.
    Exact: bf16 bit patterns are uint16; float32 bits are bf16<<16.
    """
    bits16 = np.arange(0, 1 << 16, dtype=np.uint16)
    bits32 = bits16.astype(np.uint32) << 16
    vals = bits32.view(np.float32)

    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= np.float32(a)) & (vals <= np.float32(b))]
    vals = np.unique(vals.astype(np.float32))
    return vals


# -------------------------
# Chebyshev basis utilities on [a,b]
# -------------------------


def x_to_t(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (2.0 * x - (a + b)) / (b - a)


def cheb_vander_t(t: np.ndarray, deg: int) -> np.ndarray:
    # columns: T_0(t),...,T_deg(t)
    return np.polynomial.chebyshev.chebvander(t, deg).astype(np.float64)


def cheb_eval_bf16(x: np.ndarray, a: float, b: float, c_cheb: np.ndarray) -> np.ndarray:
    """
    Evaluate q(x)=sum c_k T_k(t(x)) with bf16 rounding after each arithmetic op.
    This is a scalar model of bf16 recurrence evaluation (conservative).
    """
    x = x.astype(np.float32, copy=False)
    c_cheb = c_cheb.astype(np.float32, copy=False)
    d = c_cheb.size - 1

    t = bf16_round_f32(x_to_t(x, a, b).astype(np.float32))
    T0 = bf16_round_f32(np.ones_like(t, dtype=np.float32))
    if d == 0:
        return bf16_round_f32(c_cheb[0] * T0)

    T1 = t
    q = bf16_round_f32(c_cheb[0] * T0 + c_cheb[1] * T1)

    for k in range(2, d + 1):
        Tk = bf16_round_f32(2.0 * t * T1 - T0)
        q = bf16_round_f32(q + c_cheb[k] * Tk)
        T0, T1 = T1, Tk

    return q


# -------------------------
# LP design: maximize min g(x)=sqrt(x) q(x), with upper cap g(x) <= 1-mu
# -------------------------


@dataclass
class DesignResult:
    c_cheb: np.ndarray
    g_min_proxy: float
    mu: float


def solve_one_sided_lp(
    ell: float, deg: int, mu: float, x_proxy: np.ndarray, coef_bound: float
) -> DesignResult:
    """
    Variables: [c0..cd, m]
    Maximize m subject to:
      g_i = sqrt(x_i) * sum_k c_k T_k(t_i)
      g_i >= m
      g_i <= 1 - mu
    Optional: coefficient bounds |c_k| <= coef_bound
    """
    a, b = float(ell), 1.0
    x = x_proxy.astype(np.float64)
    t = x_to_t(x, a, b).astype(np.float64)
    V = cheb_vander_t(t, deg)  # (N, d+1)
    s = np.sqrt(x).reshape(-1, 1)
    G = s * V  # (N, d+1), so g(x)=G @ c

    nvar = (deg + 1) + 1  # c + m
    c_obj = np.zeros(nvar, dtype=np.float64)
    c_obj[-1] = -1.0  # minimize -m == maximize m

    # Inequalities A_ub @ z <= b_ub

    # 1) g_i >= m  <=>  -g_i + m <= 0
    A1 = np.hstack([-G, np.ones((G.shape[0], 1), dtype=np.float64)])
    b1 = np.zeros(G.shape[0], dtype=np.float64)

    # 2) g_i <= 1 - mu  <=>  g_i <= 1-mu
    A2 = np.hstack([G, np.zeros((G.shape[0], 1), dtype=np.float64)])
    b2 = (1.0 - mu) * np.ones(G.shape[0], dtype=np.float64)

    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])

    bounds = [(-coef_bound, coef_bound)] * (deg + 1) + [(0.0, 1.0)]  # m in [0,1]

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed (mu={mu}): {res.message}")

    z = res.x
    c_cheb = z[: deg + 1].astype(np.float64)
    m = float(z[-1])

    return DesignResult(c_cheb=c_cheb, g_min_proxy=m, mu=mu)


# -------------------------
# Verification on exhaustive bf16 set
# -------------------------


def verify_bf16_no_overshoot(
    ell: float, c_cheb: np.ndarray, mu: float
) -> Tuple[bool, float, float]:
    """
    Verify on ALL bf16 representables in [ell,1]:
      g_bf16(x) = sqrt(x) * q_bf16(x) <= 1
    Also report max_g and min_g seen in bf16 eval.
    """
    a, b = float(ell), 1.0
    xs = all_bf16_values_in_interval(a, b)
    xs = xs[xs > 0]

    q = cheb_eval_bf16(xs, a, b, c_cheb.astype(np.float32))
    g = bf16_round_f32(np.sqrt(xs.astype(np.float32)) * q)

    max_g = float(np.max(g))
    min_g = float(np.min(g))

    ok = max_g <= 1.0  # we enforce strict <= 1 in bf16 eval
    return ok, max_g, min_g


# -------------------------
# Main: proxy + exhaustive, binary search mu
# -------------------------


def build_proxy_set(ell: float, n_log: int, n_lin: int) -> np.ndarray:
    # Dense proxy grid in real numbers:
    # - log-spaced near ell to capture behavior for tiny eigenvalues
    # - lin-spaced near 1 to capture overshoot risk near the top
    a, b = float(ell), 1.0
    xs = []
    if n_log > 0:
        xs.append(np.geomspace(a, b, n_log, dtype=np.float64))
    if n_lin > 0:
        xs.append(np.linspace(a, b, n_lin, dtype=np.float64))
    x = np.unique(np.concatenate(xs))
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, required=True)
    ap.add_argument("--deg", type=int, required=True)

    ap.add_argument(
        "--mu-hi", type=float, default=0.02, help="upper bound for safety margin search"
    )
    ap.add_argument("--mu-iters", type=int, default=20)

    ap.add_argument("--proxy-log", type=int, default=5000)
    ap.add_argument("--proxy-lin", type=int, default=5000)

    ap.add_argument(
        "--include-bf16-in-proxy",
        action="store_true",
        help="add all bf16 representables in [ell,1] to the proxy constraint set",
    )

    ap.add_argument(
        "--coef-bound",
        type=float,
        default=1e4,
        help="LP bound on Chebyshev coeff magnitudes",
    )
    ap.add_argument("--out", type=str, default="bf16_safe_poly_phase1.json")
    args = ap.parse_args()

    ell = float(args.ell)
    if not (0.0 < ell < 1.0):
        raise ValueError("Need 0 < ell < 1")

    # Proxy set: continuous grid, optionally union bf16 values
    x_proxy = build_proxy_set(ell, args.proxy_log, args.proxy_lin)
    if args.include_bf16_in_proxy:
        x_bf16 = all_bf16_values_in_interval(ell, 1.0).astype(np.float64)
        x_proxy = np.unique(np.concatenate([x_proxy, x_bf16]))

    # Quick note: next bf16 after 1 is 1 + 2^{-7} = 1.0078125
    next_after_1 = 1.0 + 2.0 ** (-7)

    # Binary search minimal mu that passes exhaustive bf16 eval
    lo, hi = 0.0, float(args.mu_hi)
    best = None

    for _ in range(args.mu_iters):
        mu = 0.5 * (lo + hi)

        try:
            sol = solve_one_sided_lp(
                ell=ell,
                deg=args.deg,
                mu=mu,
                x_proxy=x_proxy,
                coef_bound=args.coef_bound,
            )
        except RuntimeError:
            # infeasible: need smaller cap
            # Here failure typically means coef bounds too small or mu too large; treat as "not feasible", move hi down.
            hi = mu
            continue

        ok, max_g, min_g = verify_bf16_no_overshoot(ell, sol.c_cheb, mu)
        if ok:
            best = (sol, max_g, min_g)
            hi = mu
        else:
            lo = mu

    if best is None:
        raise RuntimeError(
            "No feasible mu found. Try increasing --mu-hi or --coef-bound, or increase proxy density."
        )

    sol, max_g, min_g = best

    out = {
        "kind": "phase1_bf16_safe",
        "ell": ell,
        "deg": int(args.deg),
        "next_bf16_after_1": next_after_1,
        "mu_star": float(sol.mu),
        "proxy_min_g": float(sol.g_min_proxy),
        "bf16_eval_max_g": float(max_g),
        "bf16_eval_min_g": float(min_g),
        "coeff_cheb_domain_[ell,1]": sol.c_cheb.tolist(),
        "notes": {
            "invariant": "exhaustive bf16 scalar eval enforces sqrt(x)*q(x) <= 1, so x' = x*q(x)^2 <= 1",
            "proxy_set": "dense real grid (log+lin) plus optional bf16 union",
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"next bf16 after 1: {next_after_1:.7f}")
    print(f"mu* (minimal safety margin): {sol.mu:.8f}")
    print(f"proxy min g: {sol.g_min_proxy:.8f}")
    print(f"bf16 exhaustive max g: {max_g:.8f} (must be <= 1.0)")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
