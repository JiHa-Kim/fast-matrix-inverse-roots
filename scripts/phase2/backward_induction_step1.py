import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32, all_bf16_values_in_interval

def compute_pure_scalar_mapping(rho_in_max, d_choices, r=2.0, num_rho_points=50):
    """
    Sweeps rho_in and computes the max bf16 scalar evaluation error (rho_out).
    """
    rho_ins = np.linspace(0.05, rho_in_max, num_rho_points)
    
    print(f"{'rho_in':<10} | " + " | ".join([f"{f'd={d} rho_out':<15}" for d in d_choices]))
    print("-" * (13 + 18 * len(d_choices)))
    
    for rho in rho_ins:
        row = [f"{rho:<10.4f}"]
        
        for d in d_choices:
            try:
                out = design_local(
                    rho=rho,
                    deg=d,
                    basis='cheb',
                    r=r,
                    proxy_log=1000,
                    proxy_lin=2000,
                    coef_bound=1e6,
                    refine_bf16=False
                )
                rho_out = out['bf16_max_cert_err']
                row.append(f"{rho_out:<15.6f}")
            except Exception as e:
                row.append(f"{'FAIL':<15}")
                
        print(" | ".join(row))

if __name__ == "__main__":
    compute_pure_scalar_mapping(rho_in_max=0.8, d_choices=[2, 3, 4, 5], num_rho_points=20)
