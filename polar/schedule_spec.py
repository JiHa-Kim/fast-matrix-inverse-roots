from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str
    ell_in: float
    ell_out: float
    pred_kappa_after: float
    r: int = 0
    degree: int = 0
    interval_lo: float = 0.0
    interval_hi: float = 0.0
    fit_kind: str = ""
    basis_kind: str = ""
    coeffs: tuple[float, ...] = ()
    paper_coeffs: tuple[float, ...] = ()
