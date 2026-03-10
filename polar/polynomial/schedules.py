from __future__ import annotations

from typing import List

from polar.polynomial.express import paper_polar_express_coeff
from polar.polynomial.minimax import poly_inv_sqrt_coeffs_from_ell
from polar.schedule_spec import StepSpec


def _poly_step(ell: float, degree: int) -> StepSpec:
    coeffs = poly_inv_sqrt_coeffs_from_ell(degree, ell)
    sigma_min = max(coeffs.pred_sigma_min, 1e-300)
    sigma_max = max(coeffs.pred_sigma_max, sigma_min)
    ell_out = sigma_min / sigma_max
    return StepSpec(
        kind="POLY",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(sigma_max / sigma_min),
        degree=int(degree),
    )


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
                paper_coeffs=(coeffs.a, coeffs.b, coeffs.c),
            )
        )
    return out


def build_polynomial_schedule(schedule_name: str, ell: float) -> List[StepSpec] | None:
    if schedule_name == "poly16x2":
        s1 = _poly_step(ell, 16)
        s2 = _poly_step(s1.ell_out, 16)
        return [s1, s2]

    if schedule_name == "poly24x2":
        s1 = _poly_step(ell, 24)
        s2 = _poly_step(s1.ell_out, 24)
        return [s1, s2]

    if schedule_name == "pe5paper":
        return _paper_schedule(ell)

    return None
