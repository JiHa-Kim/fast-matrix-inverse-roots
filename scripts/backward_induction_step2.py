import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32, all_bf16_values_in_interval

sys.path.insert(0, os.path.abspath('.'))
from fast_iroot.eval import apply_poly_right_cheb, apply_poly_right_mono

def run_gemm_emulation(rho_in, d, n=1024, basis='cheb', num_trials=5):
    # 1. Design scalar polynomial
    out = design_local(
        rho=rho_in, deg=d, basis=basis, r=2.0, 
        proxy_log=1000, proxy_lin=2000, coef_bound=1e6, refine_bf16=False
    )
    scalar_rho_out = out['bf16_max_cert_err']
    coeffs = out['coeffs']
    
    a_dom = max(1e-6, 1.0 - rho_in)
    b_dom = 1.0 + rho_in
    
    max_hardware_rho_out = 0.0
    
    for trial in range(num_trials):
        # 2. Construct adversarial matrix
        # Generate random eigenvalues exactly spanning [1-rho_in, 1+rho_in]
        eigvals = torch.empty(n, dtype=torch.float64).uniform_(1.0 - rho_in, 1.0 + rho_in)
        # Force exact endpoints
        eigvals[0] = 1.0 - rho_in
        eigvals[1] = 1.0 + rho_in
        
        # Random orthonormal matrix for basis
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        S_exact = Q @ torch.diag(eigvals) @ Q.T
        
        # Convert to bf16
        S_bf16 = S_exact.to(torch.bfloat16)
        
        # 3. Hardware Evaluation
        
        coeffs_t = torch.tensor(coeffs, dtype=torch.bfloat16).cuda()
        S_cuda = S_bf16.cuda()
        Z_eye = torch.eye(n, dtype=torch.bfloat16).cuda()
        
        # Evaluate polynomial q(S)
        if basis == 'cheb':
            Z_update = apply_poly_right_cheb(Z_eye, S_cuda, coeffs_t, a_dom, b_dom)
        else:
            Z_update = apply_poly_right_mono(Z_eye, S_cuda, coeffs_t)
            
        Z_update = Z_update.to(torch.bfloat16) # Should already be bf16
        
        # S_new = Z_update * S * Z_update (since everything commutes)
        # To strictly mirror Z^T B Z where B is implicit, actually it's:
        # S_new = Z_update @ S_cuda @ Z_update 
        # Or more accurately, since we just have S, S_new = Z_update.T @ S_cuda @ Z_update
        
        # S_new = Z_update.T @ (S_cuda @ Z_update)
        tmp = torch.matmul(S_cuda, Z_update)
        S_new = torch.matmul(Z_update.transpose(0, 1), tmp)
        
        # Enforce structural symmetry in bf16
        S_new = 0.5 * (S_new + S_new.transpose(0, 1))
        
        # Find max eigenvalue deviation from 1.0
        S_new_f64 = S_new.cpu().to(torch.float64)
        eigs_new = torch.linalg.eigvalsh(S_new_f64)
        
        hardware_rho_out = float(torch.max(torch.abs(eigs_new - 1.0)))
        if hardware_rho_out > max_hardware_rho_out:
            max_hardware_rho_out = hardware_rho_out

    return scalar_rho_out, max_hardware_rho_out

if __name__ == "__main__":
    print(f"{'rho_in':<10} | {'deg':<4} | {'Scalar rho_out':<18} | {'Hardware rho_out':<18} | {'GEMM Noise Margin':<18}")
    print("-" * 78)
    
    for d in [2, 3, 4, 5]:
        for rho_in in [0.2, 0.4, 0.6, 0.8]:
            try:
                scalar, hw = run_gemm_emulation(rho_in, d, n=1024, num_trials=3)
                margin = hw - scalar
                print(f"{rho_in:<10.4f} | {d:<4} | {scalar:<18.6f} | {hw:<18.6f} | {margin:<18.6f}")
            except Exception as e:
                print(f"{rho_in:<10.4f} | {d:<4} | {'FAIL':<18} | {'FAIL':<18} | {'FAIL':<18}")
