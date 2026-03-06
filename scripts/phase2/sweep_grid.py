import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coeffs")))
from design_local_poly import design_local

rho = 0.1
deg = 3
basis = "cheb"
r = 2.0
coef_bound = 1e6
refine_bf16 = False

log_sizes = [0, 100, 500, 1000, 2000, 4000]
lin_sizes = [0, 100, 500, 1000, 2000, 4000]

print(f"{'proxy_log':<10} | {'proxy_lin':<10} | {'bf16_err':<15} | {'lp_delta':<15} | {'time(s)':<10}")
print("-" * 65)

best_err = float('inf')
best_grid = None

for proxy_log in log_sizes:
    for proxy_lin in lin_sizes:
        if proxy_log == 0 and proxy_lin == 0:
            continue
            
        t0 = time.time()
        out = design_local(
            rho=rho,
            deg=deg,
            basis=basis,
            r=r,
            proxy_log=proxy_log,
            proxy_lin=proxy_lin,
            coef_bound=coef_bound,
            refine_bf16=refine_bf16
        )
        t1 = time.time()
        
        bf16_err = out["bf16_max_cert_err"]
        lp_delta = out["proxy_linear_delta"]
        duration = t1 - t0
        
        print(f"{proxy_log:<10} | {proxy_lin:<10} | {bf16_err:<15.6e} | {lp_delta:<15.6e} | {duration:<10.3f}")
        
        if bf16_err < best_err:
            best_err = bf16_err
            best_grid = (proxy_log, proxy_lin)

print("-" * 65)
print(f"Best grid (without bf16 refinement): proxy_log={best_grid[0]}, proxy_lin={best_grid[1]} with err={best_err:.6e}")
