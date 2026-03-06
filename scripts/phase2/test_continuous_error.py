import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('coeffs'))
from design_local_poly import design_local
from design_bf16_poly import cheb_eval_bf16, bf16_round_f32

# Design with NO continuous proxy points (only bf16 representables)
out_sparse = design_local(0.1, 5, 'cheb', 2.0, 0, 0, 1e6, False)
coeffs_sparse = np.array(out_sparse['coeffs'], dtype=np.float32)

# Design WITH continuous proxy points
out_dense = design_local(0.1, 5, 'cheb', 2.0, 1000, 2000, 1e6, False)
coeffs_dense = np.array(out_dense['coeffs'], dtype=np.float32)

# Evaluate on a dense continuous grid
xs = np.linspace(0.9, 1.1, 100000).astype(np.float32)

# Error for sparse
q_sparse = cheb_eval_bf16(xs, 0.9, 1.1, coeffs_sparse)
S_new_sparse = xs * q_sparse**2
err_sparse = np.max(np.abs(1.0 - S_new_sparse))

# Error for dense
q_dense = cheb_eval_bf16(xs, 0.9, 1.1, coeffs_dense)
S_new_dense = xs * q_dense**2
err_dense = np.max(np.abs(1.0 - S_new_dense))

print("=== Polynomial Degree 5 ===")
print(f"Sparse Grid (bf16 only) - bf16 point error: {out_sparse['bf16_max_cert_err']:.6e}")
print(f"Sparse Grid (bf16 only) - Continuous error: {err_sparse:.6e}")
print()
print(f"Dense Grid (log+lin+bf16) - bf16 point error: {out_dense['bf16_max_cert_err']:.6e}")
print(f"Dense Grid (log+lin+bf16) - Continuous error: {err_dense:.6e}")
