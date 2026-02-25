from dataclasses import dataclass
from .precond import PrecondStats


@dataclass
class AutoPolicyConfig:
    policy: str
    n_switch: int
    rho_switch: float
    kappa_ns3_max: float
    kappa_pe2_min: float

    def __post_init__(self):
        if self.policy not in ("size_rho", "interval", "combined"):
            raise ValueError(f"Unknown auto policy: {self.policy}")
        if self.n_switch < 1:
            raise ValueError("n_switch must be >= 1")
        if self.rho_switch <= 0 or self.kappa_ns3_max <= 0 or self.kappa_pe2_min <= 0:
            raise ValueError("Thresholds must be positive")
        if (
            self.policy in ("interval", "combined")
            and self.kappa_ns3_max > self.kappa_pe2_min
        ):
            raise ValueError(
                "kappa_ns3_max cannot be strictly greater than kappa_pe2_min"
            )


def choose_auto_method(n: int, stats: PrecondStats, cfg: AutoPolicyConfig) -> str:
    # Return one of: "NS3", "PE-NS3", "PE2"
    if cfg.policy == "size_rho":
        if (n >= int(cfg.n_switch)) or (stats.rho_proxy >= float(cfg.rho_switch)):
            return "PE2"
        return "PE-NS3"

    if cfg.policy == "interval":
        if stats.kappa_proxy >= float(cfg.kappa_pe2_min):
            return "PE2"
        if stats.kappa_proxy <= float(cfg.kappa_ns3_max):
            return "NS3"
        return "PE-NS3"

    if cfg.policy == "combined":
        if (n >= int(cfg.n_switch)) or (stats.rho_proxy >= float(cfg.rho_switch)):
            return "PE2"
        if stats.kappa_proxy <= float(cfg.kappa_ns3_max):
            return "NS3"
        if stats.kappa_proxy >= float(cfg.kappa_pe2_min):
            return "PE2"
        return "PE-NS3"

    raise ValueError(f"Unknown auto policy: {cfg.policy}")
