# verify_phase2_policy.py
import argparse
import os
import sys
import torch

# Ensure local modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_iroot.eval import (
    PHASE2_TRANSITION_COEFFS,
    PHASE2_TRANSITION_RHO,
    PHASE2_TERMINAL_COEFFS,
    PHASE2_TERMINAL_RHO,
    step_phase2_local,
)


def rho_from_certificate(Z: torch.Tensor, B: torch.Tensor, mode: str) -> float:
    """
    rho(S) := ||S - I||_2 where S = Z^T B Z (symmetrized).

    mode:
      - "kernel": form S using bf16 GEMMs (includes kernel behavior and bf16 output rounding)
      - "data"  : form S in fp64 using bf16-stored values (isolates data quantization from GEMM rounding)
    """
    if mode == "kernel":
        # bf16 matmuls -> bf16 output, then upcast for eig
        S = (Z.T @ B @ Z).float()
    elif mode == "data":
        Zd = Z.double()
        Bd = B.double()
        S = Zd.T @ Bd @ Zd
    else:
        raise ValueError(f"Unknown mode: {mode}")

    S = 0.5 * (S + S.T)
    e = torch.linalg.eigvalsh(S.double())
    return float(torch.max(torch.abs(e - 1.0)))


def make_spectrum(n: int, rho: float, kind: str) -> torch.Tensor:
    lo, hi = 1.0 - rho, 1.0 + rho
    if kind == "linspace":
        return torch.linspace(lo, hi, n, dtype=torch.float64, device="cuda")
    if kind == "endpoints":
        a = torch.full((n // 2,), lo, dtype=torch.float64, device="cuda")
        b = torch.full((n - n // 2,), hi, dtype=torch.float64, device="cuda")
        return torch.cat([a, b], dim=0)
    if kind == "two_cluster":
        # 90% at hi, 10% at lo (often stresses minimax designs differently)
        k = max(1, n // 10)
        a = torch.full((k,), lo, dtype=torch.float64, device="cuda")
        b = torch.full((n - k,), hi, dtype=torch.float64, device="cuda")
        return torch.cat([a, b], dim=0)
    if kind == "random":
        u = torch.rand(n, dtype=torch.float64, device="cuda")
        return lo + (hi - lo) * u
    raise ValueError(f"Unknown spectrum kind: {kind}")


@torch.no_grad()
def verify_policy(
    n: int, trials: int, spectra: list[str], seed: int, guard: float
) -> None:
    print(
        f"Verify Phase 2 policy (bf16 kernel): n={n}, trials={trials}, spectra={spectra}"
    )
    print("-" * 80)

    rho_in = float(PHASE2_TRANSITION_RHO)
    rho_T = float(PHASE2_TERMINAL_RHO)

    # Track worst-case across all trials and spectrum constructions
    worst = {
        "init_data": 0.0,
        "init_kernel": 0.0,
        "after_T_data": 0.0,
        "after_T_kernel": 0.0,
        "after_star_data": 0.0,
        "after_star_kernel": 0.0,
        "gap_init": 0.0,
        "gap_after_T": 0.0,
        "gap_after_star": 0.0,
    }

    # Fail counters
    fail_outside_interval = 0
    fail_transition = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for t in range(trials):
        for kind in spectra:
            eigvals = make_spectrum(n, rho_in, kind=kind)
            Q, _ = torch.linalg.qr(
                torch.randn(n, n, dtype=torch.float64, device="cuda")
            )

            # Construct in fp64, then cast to bf16 (spectrum will perturb)
            B64 = Q @ torch.diag(eigvals) @ Q.T
            B = B64.to(torch.bfloat16)

            Z = torch.eye(n, dtype=torch.bfloat16, device="cuda")

            rho0_d = rho_from_certificate(Z, B, mode="data")
            rho0_k = rho_from_certificate(Z, B, mode="kernel")

            worst["init_data"] = max(worst["init_data"], rho0_d)
            worst["init_kernel"] = max(worst["init_kernel"], rho0_k)
            worst["gap_init"] = max(worst["gap_init"], max(0.0, rho0_k - rho0_d))

            # Require the measured pre-step rho to be within the polynomial's design interval
            if rho0_k > rho_in + guard:
                fail_outside_interval += 1
                continue

            # Transition step
            Z1 = step_phase2_local(
                Z, B, PHASE2_TRANSITION_RHO, PHASE2_TRANSITION_COEFFS
            )
            rho1_d = rho_from_certificate(Z1, B, mode="data")
            rho1_k = rho_from_certificate(Z1, B, mode="kernel")

            worst["after_T_data"] = max(worst["after_T_data"], rho1_d)
            worst["after_T_kernel"] = max(worst["after_T_kernel"], rho1_k)
            worst["gap_after_T"] = max(worst["gap_after_T"], max(0.0, rho1_k - rho1_d))

            if rho1_k >= rho_T:
                fail_transition += 1
                continue

            # Terminal step
            Z2 = step_phase2_local(Z1, B, PHASE2_TERMINAL_RHO, PHASE2_TERMINAL_COEFFS)
            rho2_d = rho_from_certificate(Z2, B, mode="data")
            rho2_k = rho_from_certificate(Z2, B, mode="kernel")

            worst["after_star_data"] = max(worst["after_star_data"], rho2_d)
            worst["after_star_kernel"] = max(worst["after_star_kernel"], rho2_k)
            worst["gap_after_star"] = max(
                worst["gap_after_star"], max(0.0, rho2_k - rho2_d)
            )

    total_cases = trials * len(spectra)
    print(f"PHASE2_TRANSITION_RHO (rho_in) = {rho_in:.6f}")
    print(f"PHASE2_TERMINAL_RHO   (rho_T)  = {rho_T:.6f}")
    print()

    print("Worst-case rho over all executed cases:")
    print(
        f"  init:       rho_data={worst['init_data']:.6f}, rho_kernel={worst['init_kernel']:.6f}, gap={worst['gap_init']:.6f}"
    )
    print(
        f"  after_T:    rho_data={worst['after_T_data']:.6f}, rho_kernel={worst['after_T_kernel']:.6f}, gap={worst['gap_after_T']:.6f}"
    )
    print(
        f"  after_star: rho_data={worst['after_star_data']:.6f}, rho_kernel={worst['after_star_kernel']:.6f}, gap={worst['gap_after_star']:.6f}"
    )
    print()

    print("Failure counts (informative, not proof):")
    print(
        f"  outside design interval (rho0_kernel > rho_in + guard): {fail_outside_interval}/{total_cases}"
    )
    print(
        f"  transition failed to enter terminal zone (rho1_kernel >= rho_T): {fail_transition}/{total_cases}"
    )
    print()

    # Suggested empirical knobs
    eps_hw_suggest = worst[
        "gap_after_T"
    ]  # conservative: kernel inflation relative to data at the critical boundary
    rho_plat_suggest = worst["after_star_kernel"]

    print("Suggested empirical calibration (for this n/trials/spectra/hardware):")
    print(
        f"  epsilon_hw  ~= {eps_hw_suggest:.6f}  (use as margin in Step A inequality)"
    )
    print(
        f"  rho_plat    ~= {rho_plat_suggest:.6f}  (observed post-terminal plateau bound)"
    )
    print("-" * 80)


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end Phase 2 policy verification (bf16 kernel)."
    )
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--spectra",
        type=str,
        default="endpoints,linspace,two_cluster,random",
        help="Comma-separated: endpoints,linspace,two_cluster,random",
    )
    ap.add_argument(
        "--guard",
        type=float,
        default=0.0,
        help="Allow rho0_kernel <= rho_in + guard before applying transition step.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Verification requires GPU for native bf16 GEMM.")
        sys.exit(0)

    spectra = [s.strip() for s in args.spectra.split(",") if s.strip()]
    verify_policy(
        n=args.n, trials=args.trials, spectra=spectra, seed=args.seed, guard=args.guard
    )


if __name__ == "__main__":
    main()
