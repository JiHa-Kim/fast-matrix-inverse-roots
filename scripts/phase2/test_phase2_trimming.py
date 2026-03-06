import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32, all_bf16_values_in_interval

rhos = [0.2, 0.1, 0.05, 0.02, 0.01]
degrees = [2, 3, 4, 5]
r = 2.0

print(f"{'Target rho':<12} | {'Degree':<8} | {'Theoretical FP64 err':<22} | {'Exact BF16 Eval err':<22}")
print("-" * 75)

for rho in rhos:
    for deg in degrees:
        # Design the polynomial for this rho and degree
        out = design_local(
            rho=rho,
            deg=deg,
            basis='cheb',
            r=r,
            proxy_log=1000,
            proxy_lin=2000,
            coef_bound=1e6,
            refine_bf16=False # Just look at the base minimax performance first
        )
        
        # Theoretical FP64 max error on the dense proxy set (from LP delta)
        fp64_err = out['proxy_linear_delta']
        
        # Exact BF16 max error on the actual bf16 points in the interval
        bf16_err = out['bf16_max_cert_err']
        
        print(f"{rho:<12} | {deg:<8} | {fp64_err:<22.6e} | {bf16_err:<22.6e}")
    print("-" * 75)
