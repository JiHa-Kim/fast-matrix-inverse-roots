#!/usr/bin/env python3
# design_local_poly.py
"""
Phase 2 local polynomial designer (Chebyshev basis), self-contained.

Goal (exact arithmetic):
  Find q(x) (polynomial in Chebyshev basis on x in [a,b]) that makes
    x q(x)^2 ~ 1
  which implies contraction of rho(S) = ||S-I||_2 via the exact-arithmetic map
    lambda^+ = lambda q(lambda)^2.

We solve a *linear* proxy problem via y(x) := sqrt(x) q(x):
  minimize delta_y s.t. |1 - y(x)| <= delta_y on a proxy grid.
Then, if y(x) >= 0 on [a,b], we have the rigorous implication:
  |1 - x q(x)^2| = |1 - y(x)^2| = |1-y(x)| |1+y(x)| <= delta_y (2 + delta_y).

This script reports:
  - lp_delta_y: the LP optimum delta_y on the proxy grid
  - cert_bound_from_delta_y: delta_y (2 + delta_y)
  - cert_sup_f64_grid: max |1 - x q(x)^2| on a dense float64 grid
  - cert_sup_bf16_scalar_model: max |1 - bf16( x * bf16(q^2) )| over ALL bf16 x in [a,b],
      where q is evaluated by bf16-rounded Clenshaw in float32.

Notes:
  - "bf16 scalar model" is useful but is not a GEMM model.
  - End-to-end kernel behavior must be calibrated separately.
"""

import argparse
import json
import numpy as np
from scipy.optimize import linprog


# ---------------------------
# bf16 utilities (exact enumeration + round-to-nearest-even)
# ---------------------------


def bf16_round_f32(x_f32: np.ndarray) -> np.ndarray:
    """Round float32 array to bf16 (stored as float32 values that are bf16-representable)."""
    x = x_f32.astype(np.float32, copy=False)
    u = x.view(np.uint32)

    # Round-to-nearest-even by adding bias to the lower 16 bits before truncation.
    lsb = (u >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    u_rounded = u + bias
    u_bf16 = u_rounded & np.uint32(0xFFFF0000)

    return u_bf16.view(np.float32)


def all_bf16_values_in_interval(a: float, b: float) -> np.ndarray:
    """
    Enumerate all positive bf16-representable float32 values in [a,b], assuming a>0 and b>0.

    For positive numbers, bf16 values correspond exactly to the top 16 bits of float32.
    """
    if not (a > 0.0 and b > 0.0 and a <= b):
        raise ValueError(
            "Require 0 < a <= b for exact bf16 enumeration in this helper."
        )

    a32 = bf16_round_f32(np.array([a], dtype=np.float32))[0]
    b32 = bf16_round_f32(np.array([b], dtype=np.float32))[0]

    ua = a32.view(np.uint32) >> 16
    ub = b32.view(np.uint32) >> 16
    if ua > ub:
        ua, ub = ub, ua

    codes = np.arange(int(ua), int(ub) + 1, dtype=np.uint32)
    u32 = codes << np.uint32(16)
    xs = u32.view(np.float32)
    return xs


def bf16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bf16_round_f32(a.astype(np.float32) + b.astype(np.float32))


def bf16_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bf16_round_f32(a.astype(np.float32) * b.astype(np.float32))


# ---------------------------
# Chebyshev mapping / evaluation
# ---------------------------


def x_to_t(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map x in [a,b] to t in [-1,1]."""
    return (2.0 * x - (a + b)) / (b - a)


def cheb_vander_t(t: np.ndarray, deg: int) -> np.ndarray:
    """Vandermonde for Chebyshev T_k(t), k=0..deg (float64)."""
    t = t.astype(np.float64, copy=False)
    V = np.empty((t.size, deg + 1), dtype=np.float64)
    V[:, 0] = 1.0
    if deg >= 1:
        V[:, 1] = t
    for k in range(2, deg + 1):
        V[:, k] = 2.0 * t * V[:, k - 1] - V[:, k - 2]
    return V


def cheb_eval_bf16(
    xs: np.ndarray, a: float, b: float, coeffs_f32: np.ndarray
) -> np.ndarray:
    """
    Evaluate q(x) = sum_{k=0}^deg c_k T_k(t(x)) via Clenshaw,
    rounding every elementary op to bf16 (scalar bf16 rounding model).
    """
    xs = xs.astype(np.float32, copy=False)
    coeffs = coeffs_f32.astype(np.float32, copy=False)

    t = x_to_t(xs.astype(np.float64), a, b).astype(np.float32)
    t = bf16_round_f32(t)

    deg = coeffs.shape[0] - 1
    if deg < 0:
        raise ValueError("Empty coeffs")

    # Clenshaw for Chebyshev series:
    # b_{k} = 2 t b_{k+1} - b_{k+2} + c_k, with b_{deg+1}=b_{deg+2}=0
    b_kplus1 = bf16_round_f32(np.zeros_like(t, dtype=np.float32))
    b_kplus2 = bf16_round_f32(np.zeros_like(t, dtype=np.float32))

    two_t = bf16_round_f32(2.0 * t)

    for k in range(deg, 0, -1):
        term = bf16_mul(two_t, b_kplus1)
        term = bf16_add(term, -b_kplus2)
        term = bf16_add(term, bf16_round_f32(coeffs[k]))
        b_k = term
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    # q = t*b1 - b2 + c0
    q = bf16_mul(t, b_kplus1)
    q = bf16_add(q, -b_kplus2)
    q = bf16_add(q, bf16_round_f32(coeffs[0]))
    return q


# ---------------------------
# LP design and verification
# ---------------------------


def solve_local_lp(
    a: float, b: float, deg: int, x_proxy: np.ndarray, enforce_nonneg: bool
) -> tuple[np.ndarray, float]:
    """
    Linear proxy LP on y(x) := sqrt(x) q(x).

    Variables: c_0..c_deg, delta_y
    Constraints on proxy grid:
      -delta_y <= 1 - sqrt(x) * q(x) <= delta_y
      and optionally sqrt(x)*q(x) >= 0 (to justify |1 - y^2| <= delta_y(2+delta_y)).
    """
    x = x_proxy.astype(np.float64, copy=False)
    s = np.sqrt(x).reshape(-1, 1)

    t = x_to_t(x, a, b).astype(np.float64)
    V = cheb_vander_t(t, deg)  # (m, deg+1)
    G = s * V  # y(x) = G c

    # Minimize delta_y
    c_obj = np.zeros(deg + 2, dtype=np.float64)
    c_obj[-1] = 1.0

    # |1 - Gc| <= delta  <=>  Gc - delta <= 1 and -Gc - delta <= -1
    A1 = np.hstack([G, -np.ones((G.shape[0], 1))])
    A2 = np.hstack([-G, -np.ones((G.shape[0], 1))])
    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([np.ones(G.shape[0]), -np.ones(G.shape[0])])

    if enforce_nonneg:
        # Gc >= 0  <=>  -Gc <= 0
        A3 = np.hstack([-G, np.zeros((G.shape[0], 1))])
        A_ub = np.vstack([A_ub, A3])
        b_ub = np.concatenate([b_ub, np.zeros(G.shape[0])])

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * (deg + 1) + [(0.0, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    return res.x[: deg + 1].astype(np.float64), float(res.x[-1])


def cert_sup_f64_grid(
    a: float, b: float, coeffs: np.ndarray, grid: int = 20001
) -> float:
    """Dense float64 grid sup of |1 - x q(x)^2| (exact arithmetic model)."""
    xs = np.linspace(a, b, grid, dtype=np.float64)
    t = x_to_t(xs, a, b)
    V = cheb_vander_t(t, coeffs.shape[0] - 1)
    q = V @ coeffs.astype(np.float64)
    S_new = xs * (q * q)
    return float(np.max(np.abs(1.0 - S_new)))


def cert_sup_bf16_scalar_model(a: float, b: float, coeffs: np.ndarray) -> float:
    """
    Scalar bf16 rounding model:
      q(x) evaluated with bf16-rounded Clenshaw (float32 ops + bf16 rounding),
      then S_new = bf16( x * bf16(q^2) ).
    Sup is taken over ALL bf16-representable x in [a,b].
    """
    xs = all_bf16_values_in_interval(a, b)  # float32 values, bf16-representable
    q = cheb_eval_bf16(xs, a, b, coeffs.astype(np.float32))

    q2 = bf16_mul(q, q)
    S_new = bf16_mul(xs.astype(np.float32), q2)

    return float(np.max(np.abs(1.0 - S_new).astype(np.float64)))


def design_local(rho: float, deg: int, enforce_nonneg: bool) -> dict:
    a = max(1e-6, 1.0 - rho)
    b = 1.0 + rho

    # Proxy grid: dense float grid + all bf16 representables (exact enumeration)
    x_cont = np.linspace(a, b, 3000, dtype=np.float64)
    x_bf16 = all_bf16_values_in_interval(a, b).astype(np.float64)
    x_proxy = np.unique(np.concatenate([x_cont, x_bf16]))

    coeffs, delta_y = solve_local_lp(a, b, deg, x_proxy, enforce_nonneg=enforce_nonneg)
    cert_bound = delta_y * (2.0 + delta_y) if enforce_nonneg else float("nan")

    sup_f64 = cert_sup_f64_grid(a, b, coeffs)
    sup_bf16 = cert_sup_bf16_scalar_model(a, b, coeffs)

    return {
        "rho": float(rho),
        "a_dom": float(a),
        "b_dom": float(b),
        "deg": int(deg),
        "enforce_nonneg_y": bool(enforce_nonneg),
        "lp_delta_y": float(delta_y),
        "cert_bound_from_delta_y": float(cert_bound),
        "cert_sup_f64_grid": float(sup_f64),
        "cert_sup_bf16_scalar_model": float(sup_bf16),
        "coeffs": coeffs.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 local Chebyshev polynomial designer (self-contained)."
    )
    ap.add_argument(
        "--rho", type=float, required=True, help="Radius rho for [1-rho, 1+rho]."
    )
    ap.add_argument("--deg", type=int, default=3, help="Chebyshev degree.")
    ap.add_argument("--out", type=str, required=True, help="JSON output file.")
    ap.add_argument(
        "--enforce-nonneg",
        action="store_true",
        help="Add LP constraints sqrt(x)q(x) >= 0 on the proxy grid (enables cert_bound_from_delta_y).",
    )
    args = ap.parse_args()

    out = design_local(rho=args.rho, deg=args.deg, enforce_nonneg=args.enforce_nonneg)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote: {args.out}")
    print(f"Interval: [{out['a_dom']:.6f}, {out['b_dom']:.6f}] (rho={out['rho']:.6f})")
    print(f"Degree: {out['deg']}, enforce_nonneg_y={out['enforce_nonneg_y']}")
    print(f"LP delta_y: {out['lp_delta_y']:.6g}")
    if out["enforce_nonneg_y"]:
        print(f"Cert bound from delta_y: {out['cert_bound_from_delta_y']:.6g}")
    print(f"Cert sup (float64 dense grid): {out['cert_sup_f64_grid']:.6g}")
    print(
        f"Cert sup (bf16 scalar model, all bf16 x): {out['cert_sup_bf16_scalar_model']:.6g}"
    )


if __name__ == "__main__":
    main()
