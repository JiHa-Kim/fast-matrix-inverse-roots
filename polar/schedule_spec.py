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
    coeffs: tuple[float, ...] = ()
    paper_coeffs: tuple[float, ...] = ()
