import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32, all_bf16_values_in_interval

# Design ONE polynomial for rho=0.1, d=3
out = design_local(0.1, 3, 'cheb', 2.0, 1000, 2000, 1e6, False)
coeffs = np.array(out['coeffs'], dtype=np.float32)

print("Evaluating a SINGLE polynomial (designed for rho=0.1, d=3) across different rhos:")
print(f"{'Eval rho':<12} | {'Max BF16 Err':<22}")
print("-" * 37)

for rho in [0.2, 0.1, 0.05, 0.02, 0.01]:
    # Get all bf16 points in this specific interval
    xs = all_bf16_values_in_interval(1.0 - rho, 1.0 + rho)
    
    # Evaluate the SAME polynomial
    q = cheb_eval_bf16(xs, 0.9, 1.1, coeffs)  # domain is fixed to [0.9, 1.1] based on design
    S_new = bf16_round_f32(xs * bf16_round_f32(q**2))
    err = float(np.max(np.abs(1.0 - S_new)))
    
    print(f"{rho:<12} | {err:<22.6e}")
