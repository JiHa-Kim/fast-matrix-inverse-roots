from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

from polar.ops import gram_xtx, symmetrize

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PaperPolarExpressStep:
    a: float
    b: float
    c: float


@dataclasses.dataclass(frozen=True)
class AppendixGAdditiveStep:
    ell: float
    upper: float
    cushion: float
    a: float
    b: float
    c: float
    pred_sigma_min: float
    pred_sigma_max: float
    pred_kappa_after: float


def _pe5_values(sigmas: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * sigmas + b * sigmas**3 + c * sigmas**5


def _predict_bounds(ell: float, upper: float, a: float, b: float, c: float) -> tuple[float, float]:
    sigmas = np.linspace(float(max(ell, 1e-12)), float(upper), 8193, dtype=np.float64)
    mapped = _pe5_values(sigmas, a, b, c)
    return float(np.min(mapped)), float(np.max(mapped))


def optimal_quintic(
    ell: float,
    upper: float = 1.0,
    ns_ratio_threshold: float = 1.0 - 5.0e-6,
    tol: float = 1e-15,
    max_iters: int = 64,
) -> tuple[float, float, float]:
    ell = float(ell)
    upper = float(upper)
    if not (0.0 < ell <= upper):
        raise ValueError(f"expected 0 < ell <= upper, got ell={ell}, upper={upper}")

    if ell / upper >= float(ns_ratio_threshold):
        return 15.0 / (8.0 * upper), -10.0 / (8.0 * upper**3), 3.0 / (8.0 * upper**5)

    q = (3.0 * ell + 1.0) / 4.0
    r = (ell + 3.0) / 4.0
    old_err = np.inf

    for _ in range(int(max_iters)):
        mat = np.array(
            [
                [ell, ell**3, ell**5, 1.0],
                [q, q**3, q**5, -1.0],
                [r, r**3, r**5, 1.0],
                [upper, upper**3, upper**5, -1.0],
            ],
            dtype=np.float64,
        )
        rhs = np.ones(4, dtype=np.float64)
        a, b, c, err = np.linalg.solve(mat, rhs)
        if abs(old_err - err) <= float(tol):
            return float(a), float(b), float(c)

        disc = 9.0 * b * b - 20.0 * a * c
        if disc <= 0.0:
            return float(a), float(b), float(c)
        root = float(np.sqrt(disc))
        q_next = float(np.sqrt(max((-3.0 * b + root) / (10.0 * c), 0.0)))
        r_next = float(np.sqrt(max((-3.0 * b - root) / (10.0 * c), 0.0)))
        q, r = min(q_next, r_next), max(q_next, r_next)
        old_err = err

    return float(a), float(b), float(c)


@functools.lru_cache(maxsize=256)
def additive_appendix_g_composition(
    ell_key: float,
    num_iters: int,
    cushion: float = 0.02407327424182761,
    safety_factor: float = 1.0,
) -> tuple[AppendixGAdditiveStep, ...]:
    ell = float(min(max(ell_key, 1e-12), 1.0))
    lower = ell
    upper = 1.0
    out: list[AppendixGAdditiveStep] = []

    for _ in range(int(num_iters)):
        fit_lower = max(lower, float(cushion) * upper)
        a, b, c = optimal_quintic(fit_lower, upper)
        p_lower = a * lower + b * lower**3 + c * lower**5
        p_upper = a * upper + b * upper**3 + c * upper**5
        recenter = 2.0 / (p_lower + p_upper)
        a *= recenter
        b *= recenter
        c *= recenter

        if safety_factor != 1.0:
            a /= float(safety_factor)
            b /= float(safety_factor) ** 3
            c /= float(safety_factor) ** 5

        pred_sigma_min, pred_sigma_max = _predict_bounds(lower, upper, a, b, c)
        pred_kappa_after = pred_sigma_max / max(pred_sigma_min, 1e-300)
        out.append(
            AppendixGAdditiveStep(
                ell=float(lower),
                upper=float(upper),
                cushion=float(cushion),
                a=float(a),
                b=float(b),
                c=float(c),
                pred_sigma_min=float(pred_sigma_min),
                pred_sigma_max=float(pred_sigma_max),
                pred_kappa_after=float(pred_kappa_after),
            )
        )
        lower = p_lower * recenter
        upper = p_upper * recenter

    return tuple(out)


def additive_appendix_g_coeff(
    ell: float,
    step_idx: int,
    cushion: float = 0.02407327424182761,
    safety_factor: float = 1.0,
) -> AppendixGAdditiveStep:
    idx = max(int(step_idx), 0)
    ell_key = float(f"{float(ell):.12e}")
    steps = additive_appendix_g_composition(
        ell_key=ell_key,
        num_iters=idx + 1,
        cushion=float(cushion),
        safety_factor=float(safety_factor),
    )
    return steps[idx]


_PE5_PAPER_COEFFS: tuple[PaperPolarExpressStep, ...] = (
    PaperPolarExpressStep(8.28721201814563, -23.595886519098837, 17.300387312530933),
    PaperPolarExpressStep(4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    PaperPolarExpressStep(3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    PaperPolarExpressStep(3.3184196573706015, -2.488488024314874, 0.51004894012372),
    PaperPolarExpressStep(2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    PaperPolarExpressStep(1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    PaperPolarExpressStep(1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    PaperPolarExpressStep(1.875, -1.25, 0.375),
)


def paper_polar_express_coeff(step_idx: int) -> PaperPolarExpressStep:
    idx = min(max(int(step_idx), 0), len(_PE5_PAPER_COEFFS) - 1)
    return _PE5_PAPER_COEFFS[idx]


@torch.no_grad()
def polar_express_deg5_step_matrix_only(
    S: Tensor,
    a: float,
    b: float,
    c: float,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    S_work = symmetrize(S.to(dtype=matmul_dtype))
    n = S_work.shape[0]
    I = torch.eye(n, device=S_work.device, dtype=matmul_dtype)
    S2 = symmetrize(S_work @ S_work)
    Q = symmetrize(float(a) * I + float(b) * S_work + float(c) * S2)
    if not torch.isfinite(Q).all():
        raise RuntimeError("non-finite polar express step")
    return Q, 0.0


@torch.no_grad()
def polar_express_paper5_step_matrix_only(
    S: Tensor,
    coeffs: PaperPolarExpressStep,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    return polar_express_deg5_step_matrix_only(
        S=S,
        a=coeffs.a,
        b=coeffs.b,
        c=coeffs.c,
        matmul_dtype=matmul_dtype,
    )


@torch.no_grad()
def polar_express_paper_fro_scale(
    X: Tensor,
    safety: float = 1.01,
    eps: float = 1e-7,
) -> tuple[Tensor, float]:
    fro = torch.linalg.matrix_norm(X.float(), ord="fro").clamp_min(float(eps))
    scale = float(safety) * fro
    X_scaled = X / scale.to(dtype=X.dtype)
    return X_scaled, float(scale.item())


@torch.no_grad()
def polar_express_aol_scale(
    X: Tensor,
    accum_dtype: torch.dtype,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    S = gram_xtx(X, accum_dtype)
    s = torch.rsqrt(S.abs().sum(dim=-1).clamp_min(float(eps)))
    X_scaled = X * s.unsqueeze(0).to(dtype=X.dtype)
    return X_scaled, s
