from __future__ import annotations

from typing import List

from polar.polynomial.express import additive_appendix_g_composition, paper_polar_express_coeff
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


def build_polynomial_schedule(schedule_name: str, ell: float) -> List[StepSpec] | None:
    if schedule_name == "pe5add":
        return _appendix_g_additive_schedule(ell)

    if schedule_name == "pe5paper":
        return _paper_schedule(ell)

    return None
