import sys
import os
import torch
import numpy as np

# Ensure local modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_iroot.eval import (
    PHASE2_TRANSITION_COEFFS,
    PHASE2_TRANSITION_RHO,
    PHASE2_TERMINAL_COEFFS,
    PHASE2_TERMINAL_RHO,
    step_phase2_local
)

def verify_policy(n=1024):
    print(f"Verifying Phase 2 Local Protocol (N={n}, bf16)")
    print("-" * 60)
    
    # 1. Start with a "Phase 1 Output" matrix
    # Phase 1 typically compresses spectrum to [0.5, 1.0] or better.
    # We'll use a worst-case scenario for the transition step: rho = 0.76.
    rho_start = 0.76
    eigvals = torch.linspace(1.0 - rho_start, 1.0 + rho_start, n, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    S_start = (Q @ torch.diag(eigvals) @ Q.T).to(torch.bfloat16).cuda()
    
    # We treat S_start as Z^T B Z where Z=I initially for this verification.
    B = S_start
    Z = torch.eye(n, dtype=torch.bfloat16, device='cuda')
    
    def get_rho(Z, B):
        S = Z.T @ B @ Z
        e = torch.linalg.eigvalsh(S.to(torch.float64))
        return float(torch.max(torch.abs(e - 1.0)))

    rho0 = get_rho(Z, B)
    print(f"Initial Phase 1 Output rho: {rho0:.6f}")

    # 2. Apply Transition Step (P2-A)
    # Designed for rho = 0.7653
    Z = step_phase2_local(Z, B, PHASE2_TRANSITION_RHO, PHASE2_TRANSITION_COEFFS)
    rho1 = get_rho(Z, B)
    print(f"After Transition Step (rho_in={PHASE2_TRANSITION_RHO}): {rho1:.6f}")
    
    # 3. Apply Terminal Step (P2-B)
    # Designed for rho = 0.0816
    Z = step_phase2_local(Z, B, PHASE2_TERMINAL_RHO, PHASE2_TERMINAL_COEFFS)
    rho2 = get_rho(Z, B)
    print(f"After Terminal Step (rho_in={PHASE2_TERMINAL_RHO}): {rho2:.6f}")
    
    # The noise floor is ~0.0078
    if rho2 <= 0.008:
        print("\nSUCCESS: Convergence to hardware noise floor verified.")
    else:
        print("\nWARNING: Convergence did not reach expected noise floor.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Verification requires GPU for native bf16 GEMM.")
        sys.exit(0)
    verify_policy()
