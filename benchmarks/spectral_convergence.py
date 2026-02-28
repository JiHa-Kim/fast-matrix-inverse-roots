import torch
import math
from typing import List, Tuple
from fast_iroot.coeffs import build_pe_schedules, _quad_coeffs
from fast_iroot.diagnostics import analyze_spectral_convergence, format_spectral_report, SpectralStepStats

def run_diagnostic_iteration(
    A_norm: torch.Tensor,
    abc_t: List[Tuple[float, float, float]],
    p_val: int,
) -> List[SpectralStepStats]:
    Y = A_norm.clone()
    stats = []
    
    # Step 0: Initial spectrum
    stats.append(analyze_spectral_convergence(Y, 0))
    
    for t, (a, b, c) in enumerate(abc_t):
        # B = aI + bY + cY^2
        B = a * torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype) + b * Y
        if abs(c) > 1e-9:
            B = B + c * (Y @ Y)
        
        # Update Y based on coupled PE rules
        if p_val == 1:
            Y = B @ Y
        elif p_val == 2:
            Y = B @ Y @ B
        else:
            Bp = torch.matrix_power(B, p_val)
            Y = Bp @ Y
            
        # Symmetrize to maintain real eigenvalues in simulation
        Y = 0.5 * (Y + Y.mT)
        stats.append(analyze_spectral_convergence(Y, t + 1))
        
    return stats

def aggregate_worst_case(all_stats: List[List[SpectralStepStats]]) -> List[SpectralStepStats]:
    num_steps = len(all_stats[0])
    worst_stats = []
    for step_idx in range(num_steps):
        rho_max = max(trial[step_idx].rho_residual for trial in all_stats)
        c90_min = min(trial[step_idx].clustering_90 for trial in all_stats)
        c99_min = min(trial[step_idx].clustering_99 for trial in all_stats)
        min_eig = min(trial[step_idx].min_eig for trial in all_stats)
        max_eig = max(trial[step_idx].max_eig for trial in all_stats)
        mean_eig = sum(trial[step_idx].mean_eig for trial in all_stats) / len(all_stats)
        
        worst_stats.append(SpectralStepStats(
            step=step_idx,
            min_eig=min_eig,
            max_eig=max_eig,
            mean_eig=mean_eig,
            std_eig=0.0, # Not aggregated
            rho_residual=rho_max,
            clustering_90=c90_min,
            clustering_99=c99_min
        ))
    return worst_stats

def main():
    n = 1024
    trials = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    p_val = 2
    
    print(f"# Spectral Convergence Analysis (n={n}, p={p_val}, trials={trials})")
    print(f"Target Spectral Interval: [0.05, 1.0]")
    print("Reporting WORST-CASE metrics across all trials.")
    print()

    # Get schedules
    pe_quad_t, _ = build_pe_schedules(
        l_target=0.05, device=device, coeff_mode="precomputed",
        coeff_seed=0, coeff_safety=1.0, coeff_no_final_safety=False, p_val=p_val
    )
    abc_pe = _quad_coeffs(pe_quad_t)
    abc_ns = [(1.5, -0.5, 0.0)] * len(abc_pe)

    all_pe_stats = []
    all_ns_stats = []

    for trial in range(trials):
        # Create an SPD matrix with a challenging spectrum [0.05, 1.0]
        e = torch.linspace(0.05, 1.0, steps=n, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))
        A_norm = (Q * e.unsqueeze(0)) @ Q.mT
        
        all_pe_stats.append(run_diagnostic_iteration(A_norm, abc_pe, p_val))
        all_ns_stats.append(run_diagnostic_iteration(A_norm, abc_ns, p_val))

    worst_pe = aggregate_worst_case(all_pe_stats)
    worst_ns = aggregate_worst_case(all_ns_stats)

    print("## Method: PE-Quad (Production)")
    print(format_spectral_report(worst_pe))
    print()
    
    print("## Method: Newton-Schulz (Baseline)")
    print(format_spectral_report(worst_ns))

if __name__ == "__main__":
    main()
