from __future__ import annotations

from typing import List

from polar.polynomial.express import additive_appendix_g_composition, paper_polar_express_coeff
from polar.polynomial.minimax import poly_sigma_map_coeffs_from_ell
from polar.rational.dwh_tuned_fp32 import get_tuned_dwh_coeffs_fp32
from polar.schedule_spec import StepSpec


def _appendix_g_additive_schedule(ell: float, steps: int = 5) -> List[StepSpec]:
    composition = additive_appendix_g_composition(float(f"{float(ell):.12e}"), int(steps))
    out: List[StepSpec] = []
    for step in composition:
        sigma_min = max(step.pred_sigma_min, 1e-300)
        sigma_max = max(step.pred_sigma_max, sigma_min)
        out.append(
            StepSpec(
                kind="PEADD5",
                ell_in=float(step.ell),
                ell_out=float(sigma_min / sigma_max),
                pred_kappa_after=float(step.pred_kappa_after),
                coeffs=(step.a, step.b, step.c),
            )
        )
    return out


def _paper_schedule(ell: float, steps: int = 5) -> List[StepSpec]:
    sigma_lo = max(float(ell), 1e-8)
    out: List[StepSpec] = []
    for i in range(steps):
        coeffs = paper_polar_express_coeff(i)
        out.append(
            StepSpec(
                kind="PEPAPER5",
                ell_in=float(sigma_lo),
                ell_out=float(sigma_lo),
                pred_kappa_after=float(1.0 / max(sigma_lo, 1e-300)),
                coeffs=(coeffs.a, coeffs.b, coeffs.c),
                paper_coeffs=(coeffs.a, coeffs.b, coeffs.c),
            )
        )
    return out


def _tuned_dwh_then_sigma_map(
    ell: float,
    rational_steps: int,
    poly_degrees: list[int],
    cushion: float = 1.0,
    method: str = "lawson",
    basis_kind: str = "chebyshev",
) -> List[StepSpec]:
    curr_ell = float(ell)
    out: List[StepSpec] = []
    for _ in range(int(rational_steps)):
        a, b, c = get_tuned_dwh_coeffs_fp32(curr_ell)
        next_ell = curr_ell * (a + b * curr_ell * curr_ell) / (1.0 + c * curr_ell * curr_ell)
        out.append(
            StepSpec(
                kind="DWH_TUNED_FP32",
                ell_in=float(curr_ell),
                ell_out=float(next_ell),
                pred_kappa_after=float(1.0 / max(next_ell, 1e-300)),
                r=1,
            )
        )
        curr_ell = float(next_ell)

    for degree in poly_degrees:
        fit_ell = max(min(curr_ell * float(cushion), 1.0 - 1e-6), 1e-6)
        coeffs = poly_sigma_map_coeffs_from_ell(int(degree), fit_ell, method=method, basis_kind=basis_kind)
        next_ell = float(coeffs.pred_sigma_min / max(coeffs.pred_sigma_max, 1e-300))
        out.append(
            StepSpec(
                kind="POLY_SIGMA_MAP",
                ell_in=float(curr_ell),
                ell_out=float(next_ell),
                pred_kappa_after=float(1.0 / max(next_ell, 1e-300)),
                degree=int(coeffs.degree),
                interval_lo=float(coeffs.interval_lo),
                interval_hi=float(coeffs.interval_hi),
                fit_kind=str(coeffs.fit_kind),
                basis_kind=str(coeffs.basis_kind),
                coeffs=tuple(float(v) for v in coeffs.coeffs),
            )
        )
        curr_ell = next_ell

    return out


def build_polynomial_schedule(schedule_name: str, ell: float) -> List[StepSpec] | None:
    if schedule_name == "pe5add":
        return _appendix_g_additive_schedule(ell)

    if schedule_name == "pe5paper":
        return _paper_schedule(ell)

    if schedule_name == "dwh3_sigma3x2":
        return _tuned_dwh_then_sigma_map(ell, rational_steps=3, poly_degrees=[3, 3])

    if schedule_name == "dwh3_sigma3x3":
        return _tuned_dwh_then_sigma_map(ell, rational_steps=3, poly_degrees=[3, 3, 3])

    if schedule_name in {"dwh4_cubic", "dwh4_remez_cubic", "dwh4_remez_cubic_monomial", "dwh4_sigma3", "dwh4_sigma3_remez_mono"}:
        return _tuned_dwh_then_sigma_map(
            ell,
            rational_steps=4,
            poly_degrees=[3],
            cushion=0.95,
            method="remez",
            basis_kind="monomial",
        )

    if schedule_name in {"dwh4_cubic_cheb", "dwh4_remez_cubic_chebyshev", "dwh4_sigma3_remez_cheb"}:
        return _tuned_dwh_then_sigma_map(
            ell,
            rational_steps=4,
            poly_degrees=[3],
            cushion=0.95,
            method="remez",
            basis_kind="chebyshev",
        )

    if schedule_name == "dwh4_sigma2x2":
        return _tuned_dwh_then_sigma_map(ell, rational_steps=4, poly_degrees=[2, 2], cushion=0.95)

    return None
