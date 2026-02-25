from dataclasses import dataclass
from .precond import PrecondStats


@dataclass
class AutoPolicyConfig:
    policy: str
    n_switch: int
    rho_switch: float
    kappa_ns3_max: float
    kappa_pe2_min: float


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
