import dataclasses
from typing import Dict, Any


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_cert: float
    final_kO_exact: float
    final_kO_pred: float
    steps: int
    guards: int
    ms_total: float
    ms_details: Dict[str, float]
