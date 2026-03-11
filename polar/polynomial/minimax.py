from __future__ import annotations

import dataclasses
import functools
from typing import Tuple

import numpy as np
import torch
from numpy.polynomial import Chebyshev, Polynomial

try:
    import mpmath as mp
except Exception:
    mp = None

from polar.ops import symmetrize

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PolyInvSqrtCoeffs:
    degree: int
    ell: float
    interval_lo: float
    interval_hi: float
    coeffs: Tuple[float, ...]
    max_rel_err: float
    pred_sigma_min: float
    pred_sigma_max: float
    fit_kind: str = "inv_sqrt"
    basis_kind: str = "chebyshev"


def _cheb_basis_matrix(xs: np.ndarray, degree: int) -> np.ndarray:
    basis = np.empty((xs.shape[0], degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree == 0:
        return basis
    basis[:, 1] = xs
    for k in range(2, degree + 1):
        basis[:, k] = 2.0 * xs * basis[:, k - 1] - basis[:, k - 2]
    return basis


def _scalar_interval_bounds(coeffs: np.ndarray, ell: float) -> tuple[float, float, float]:
    sigmas = np.linspace(float(max(ell, 1e-6)), 1.0, 4097, dtype=np.float64)
    xs = sigmas * sigmas
    lo = float(ell * ell)
    hi = 1.0
    mid = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    ts = (xs - mid) / radius
    T = _cheb_basis_matrix(ts, coeffs.shape[0] - 1)
    inv_sqrt = T @ coeffs
    sigma_out = sigmas * inv_sqrt
    rel = np.abs(xs * inv_sqrt * inv_sqrt - 1.0)
    return float(np.min(sigma_out)), float(np.max(sigma_out)), float(np.max(rel))


def _scaled_cheb_basis(xs: np.ndarray, lo: float, hi: float, degree: int) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    if radius <= 0.0:
        raise ValueError("interval must have positive width")
    ts = (xs - mid) / radius
    return _cheb_basis_matrix(ts, degree)


def _monomial_basis(xs: np.ndarray, degree: int) -> np.ndarray:
    basis = np.empty((xs.shape[0], degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    for k in range(1, degree + 1):
        basis[:, k] = basis[:, k - 1] * xs
    return basis


def _basis_matrix(xs: np.ndarray, lo: float, hi: float, degree: int, basis_kind: str) -> np.ndarray:
    if basis_kind == "chebyshev":
        return _scaled_cheb_basis(xs, lo, hi, degree)
    if basis_kind == "monomial":
        return _monomial_basis(xs, degree)
    raise ValueError(f"unsupported basis_kind: {basis_kind}")


def _poly_values(coeffs: np.ndarray, xs: np.ndarray, lo: float, hi: float, basis_kind: str) -> np.ndarray:
    return _basis_matrix(xs, lo, hi, coeffs.shape[0] - 1, basis_kind) @ coeffs


def _sigma_map_bounds(
    coeffs: np.ndarray,
    ell: float,
    basis_kind: str,
) -> tuple[float, float, float]:
    sigmas = np.linspace(float(max(ell, 1e-6)), 1.0, 4097, dtype=np.float64)
    xs = sigmas * sigmas
    lo = float(ell * ell)
    hi = 1.0
    q_vals = _poly_values(coeffs, xs, lo, hi, basis_kind)
    sigma_out = sigmas * q_vals
    rel = np.abs(xs * q_vals * q_vals - 1.0)
    return float(np.min(sigma_out)), float(np.max(sigma_out)), float(np.max(rel))


def _lawson_sigma_map_fit(
    degree: int,
    ell: float,
    basis_kind: str = "chebyshev",
    grid_size: int = 4097,
    iters: int = 20,
) -> np.ndarray:
    lo = float(max(ell * ell, 1e-12))
    hi = 1.0
    xs = np.linspace(lo, hi, int(grid_size), dtype=np.float64)
    sqrt_xs = np.sqrt(xs)
    basis = _basis_matrix(xs, lo, hi, int(degree), basis_kind)
    design = basis * sqrt_xs[:, None]
    target = np.ones_like(xs)
    weights = np.ones_like(xs)
    coeffs = np.zeros(int(degree) + 1, dtype=np.float64)

    for _ in range(int(iters)):
        weighted_design = design * weights[:, None]
        weighted_target = target * weights
        coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        residual = design @ coeffs - target
        abs_residual = np.maximum(np.abs(residual), 1e-12)
        new_weights = np.sqrt(abs_residual / np.max(abs_residual))
        weights = np.maximum(0.1 * weights + 0.9 * new_weights, 1e-6)

    return coeffs


def _initial_remez_nodes(lo: float, hi: float, degree: int) -> np.ndarray:
    k = np.arange(degree + 2, dtype=np.float64)
    ts = np.cos(np.pi * k / (degree + 1))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * ts)


def _select_alternating_extrema(xs: np.ndarray, residual: np.ndarray, count: int) -> np.ndarray:
    abs_res = np.abs(residual)
    candidates = [0]
    for i in range(1, residual.shape[0] - 1):
        if abs_res[i] >= abs_res[i - 1] and abs_res[i] >= abs_res[i + 1]:
            candidates.append(i)
    candidates.append(residual.shape[0] - 1)
    candidate_arr = np.array(sorted(set(candidates)), dtype=np.int64)
    ranked = candidate_arr[np.argsort(abs_res[candidate_arr])[::-1]]
    chosen = sorted(int(i) for i in ranked[:count])

    changed = True
    while changed:
        changed = False
        signs = [1 if residual[i] >= 0.0 else -1 for i in chosen]
        for j in range(len(chosen) - 1):
            if signs[j] == signs[j + 1]:
                if abs_res[chosen[j]] < abs_res[chosen[j + 1]]:
                    del chosen[j]
                else:
                    del chosen[j + 1]
                changed = True
                break

    while len(chosen) < count:
        added = False
        for idx in ranked:
            idx = int(idx)
            if idx in chosen:
                continue
            trial = sorted(chosen + [idx])
            signs = [1 if residual[i] >= 0.0 else -1 for i in trial]
            ok = True
            for j in range(len(signs) - 1):
                if signs[j] == signs[j + 1]:
                    ok = False
                    break
            if ok:
                chosen = trial
                added = True
                break
        if not added:
            break

    while len(chosen) < count:
        for idx in candidate_arr:
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) == count:
                    break
        chosen.sort()

    return xs[np.array(chosen[:count], dtype=np.int64)]


def _remez_sigma_map_fit(
    degree: int,
    ell: float,
    basis_kind: str = "chebyshev",
    grid_size: int = 8193,
    max_iters: int = 20,
    tol: float = 1e-12,
) -> np.ndarray:
    lo = float(max(ell * ell, 1e-12))
    hi = 1.0
    xs_dense = np.linspace(lo, hi, int(grid_size), dtype=np.float64)
    sqrt_dense = np.sqrt(xs_dense)
    nodes = _initial_remez_nodes(lo, hi, int(degree))
    last_err = np.inf

    coeffs = np.zeros(int(degree) + 1, dtype=np.float64)
    for _ in range(int(max_iters)):
        basis_nodes = _basis_matrix(nodes, lo, hi, int(degree), basis_kind)
        design = basis_nodes * np.sqrt(nodes)[:, None]
        alt = ((-1.0) ** np.arange(nodes.shape[0], dtype=np.float64)).reshape(-1, 1)
        system = np.concatenate([design, alt], axis=1)
        rhs = np.ones(nodes.shape[0], dtype=np.float64)
        sol = np.linalg.solve(system, rhs)
        coeffs = sol[:-1]
        err = abs(float(sol[-1]))
        values = _poly_values(coeffs, xs_dense, lo, hi, basis_kind)
        residual = sqrt_dense * values - 1.0
        if abs(err - last_err) <= float(tol) * max(1.0, err):
            break
        last_err = err
        nodes = _select_alternating_extrema(xs_dense, residual, int(degree) + 2)
    return coeffs


@functools.lru_cache(maxsize=512)
def _poly_coeffs_cached(degree: int, ell_key: float, dps: int, recenter: bool) -> PolyInvSqrtCoeffs:
    if mp is None:
        raise RuntimeError("mpmath is required for polynomial coefficients")

    degree = int(degree)
    ell = float(min(max(ell_key, 1e-6), 1.0 - 1e-6))
    lo = float(ell * ell)
    hi = 1.0
    mp.mp.dps = int(dps)

    power_hi_to_lo, _err = mp.chebyfit(lambda x: x ** (-mp.mpf("0.5")), [lo, hi], degree + 1, error=True)
    power_lo_to_hi = np.array([float(v) for v in reversed(power_hi_to_lo)], dtype=np.float64)
    cheb = Polynomial(power_lo_to_hi).convert(kind=Chebyshev, domain=[lo, hi], window=[-1.0, 1.0])
    coeffs = np.array(cheb.coef, dtype=np.float64)

    pred_sigma_min, pred_sigma_max, max_rel_err = _scalar_interval_bounds(coeffs, ell)
    if recenter:
        scale = 2.0 / max(pred_sigma_min + pred_sigma_max, 1e-300)
        coeffs = coeffs * scale
        pred_sigma_min, pred_sigma_max, max_rel_err = _scalar_interval_bounds(coeffs, ell)
    return PolyInvSqrtCoeffs(
        degree=degree,
        ell=ell,
        interval_lo=lo,
        interval_hi=hi,
        coeffs=tuple(float(v) for v in coeffs),
        max_rel_err=float(max_rel_err),
        pred_sigma_min=float(pred_sigma_min),
        pred_sigma_max=float(pred_sigma_max),
    )


def poly_inv_sqrt_coeffs_from_ell(
    degree: int,
    ell: float,
    dps: int = 100,
    recenter: bool = False,
) -> PolyInvSqrtCoeffs:
    ell_key = float(f"{float(ell):.12e}")
    return _poly_coeffs_cached(int(degree), ell_key, int(dps), bool(recenter))


@functools.lru_cache(maxsize=1024)
def _poly_sigma_map_coeffs_cached(degree: int, ell_key: float, method: str, basis_kind: str) -> PolyInvSqrtCoeffs:
    degree = int(degree)
    ell = float(min(max(ell_key, 1e-6), 1.0 - 1e-6))
    lo = float(ell * ell)
    hi = 1.0
    if method == "lawson":
        coeffs = _lawson_sigma_map_fit(degree=degree, ell=ell, basis_kind=basis_kind)
    elif method == "remez":
        coeffs = _remez_sigma_map_fit(degree=degree, ell=ell, basis_kind=basis_kind)
    else:
        raise ValueError(f"unsupported sigma-map fit method: {method}")
    pred_sigma_min, pred_sigma_max, max_rel_err = _sigma_map_bounds(coeffs, ell, basis_kind)
    return PolyInvSqrtCoeffs(
        degree=degree,
        ell=ell,
        interval_lo=lo,
        interval_hi=hi,
        coeffs=tuple(float(v) for v in coeffs),
        max_rel_err=float(max_rel_err),
        pred_sigma_min=float(pred_sigma_min),
        pred_sigma_max=float(pred_sigma_max),
        fit_kind=f"sigma_map_{method}",
        basis_kind=basis_kind,
    )


def poly_sigma_map_coeffs_from_ell(
    degree: int,
    ell: float,
    method: str = "lawson",
    basis_kind: str = "chebyshev",
) -> PolyInvSqrtCoeffs:
    ell_key = float(f"{float(ell):.12e}")
    return _poly_sigma_map_coeffs_cached(int(degree), ell_key, str(method), str(basis_kind))


@torch.no_grad()
def chebyshev_clenshaw_matrix(
    A: Tensor,
    coeffs: Tuple[float, ...],
    interval_lo: float,
    interval_hi: float,
    out_dtype: torch.dtype,
) -> Tensor:
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=out_dtype)
    lo = float(interval_lo)
    hi = float(interval_hi)
    mid = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    if radius <= 0.0:
        raise ValueError("interval must have positive width")

    work = symmetrize(A.to(dtype=out_dtype))
    T = symmetrize((work - mid * I) / radius)
    zeros = torch.zeros_like(T)
    b_kplus1 = zeros
    b_kplus2 = zeros

    for ck in reversed(coeffs[1:]):
        b_k = symmetrize(2.0 * (T @ b_kplus1) - b_kplus2 + float(ck) * I)
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    return symmetrize(T @ b_kplus1 - b_kplus2 + float(coeffs[0]) * I)


@torch.no_grad()
def monomial_matrix_poly(
    A: Tensor,
    coeffs: Tuple[float, ...],
    out_dtype: torch.dtype,
) -> Tensor:
    work = symmetrize(A.to(dtype=out_dtype))
    n = work.shape[0]
    I = torch.eye(n, device=A.device, dtype=out_dtype)
    result = float(coeffs[-1]) * I
    for ck in reversed(coeffs[:-1]):
        result = symmetrize(result @ work + float(ck) * I)
    return result


@torch.no_grad()
def poly_step_matrix_only(
    S: Tensor,
    coeffs: PolyInvSqrtCoeffs,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    if not np.isfinite(coeffs.max_rel_err):
        raise RuntimeError("polynomial fit reported non-finite residual")
    if coeffs.fit_kind == "inv_sqrt" and coeffs.max_rel_err > 0.25:
        raise RuntimeError(
            f"polynomial inverse-sqrt fit is unstable for ell={coeffs.ell:.3e}, degree={coeffs.degree}"
        )
    if coeffs.fit_kind == "sigma_map" and coeffs.pred_sigma_min <= 0.0:
        raise RuntimeError(
            f"polynomial sigma-map fit is unstable for ell={coeffs.ell:.3e}, degree={coeffs.degree}"
        )
    if coeffs.basis_kind == "chebyshev":
        Q = chebyshev_clenshaw_matrix(
            S,
            coeffs.coeffs,
            interval_lo=coeffs.interval_lo,
            interval_hi=coeffs.interval_hi,
            out_dtype=matmul_dtype,
        )
    elif coeffs.basis_kind == "monomial":
        Q = monomial_matrix_poly(
            S,
            coeffs.coeffs,
            out_dtype=matmul_dtype,
        )
    else:
        raise RuntimeError(f"unsupported polynomial basis: {coeffs.basis_kind}")
    if not torch.isfinite(Q).all():
        raise RuntimeError("non-finite polynomial Clenshaw evaluation")
    return Q, 0.0


@torch.no_grad()
def newton_schulz_inv_sqrt_matrix_only(
    S: Tensor,
    steps: int,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=matmul_dtype)
    X = I.clone()
    A = symmetrize(S.to(dtype=matmul_dtype))
    for _ in range(int(steps)):
        X2 = symmetrize(X @ X)
        X = symmetrize(0.5 * (X @ (3.0 * I - A @ X2)))
        if not torch.isfinite(X).all():
            raise RuntimeError("non-finite Newton-Schulz inverse-sqrt iterate")
    return X, 0.0
