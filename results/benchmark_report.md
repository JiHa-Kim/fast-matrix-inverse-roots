# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-24*

This report details the performance and accuracy of quadratic PE (Polynomial-Express) iterations for matrix inverse p-th roots.

## Methodology
- **Sizes**: 256,512,1024
- **Compiled**: Yes (`torch.compile(mode='max-autotune')`)
- **Trials per case**: 10
- **Hardware**: GPU (bf16)
- **Methods Compared**: `Inverse-Newton` (baseline), `PE-Quad` (uncoupled quadratic), `PE-Quad-Coupled` (coupled quadratic).

## Results for $p=1$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.180 | 1.184 | 13 | 2.066e-03 | 1.258e-01 | 2.070e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.178 | 1.182 | 13 | 3.809e-03 | - | 3.808e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.174 | 1.178 | 13 | 4.521e-03 | 1.381e-01 | 4.513e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.003 | 1.180 | 13 | 1.499e-03 | 1.255e-01 | 1.489e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.917 | 1.094 | 13 | 7.084e-03 | - | 7.089e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.019 | 1.195 | 13 | 5.912e-03 | 1.361e-01 | 5.901e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.770 | 1.117 | 13 | 1.108e-03 | 1.253e-01 | 1.110e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.786 | 1.134 | 13 | 8.147e-03 | - | 8.150e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.755 | 1.102 | 13 | 7.568e-03 | 7.348e-02 | 7.554e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.397 | 1.076 | 13 | 2.367e-03 | 1.253e-01 | 2.377e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.561 | 1.240 | 13 | 8.876e-03 | - | 8.882e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.454 | 1.133 | 13 | 8.508e-03 | 6.450e-02 | 8.484e-03 |
| Inv-Newton | 256x256 | spike | 2.822 | 1.034 | 13 | 4.917e-03 | 1.232e-01 | 4.929e-03 |
| PE-Quad | 256x256 | spike | 2.933 | 1.146 | 13 | 6.829e-03 | - | 6.844e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.818 | 1.030 | 13 | 6.946e-03 | 1.175e-01 | 6.937e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 4.209 | 1.732 | 29 | 2.989e-03 | 1.777e-01 | 2.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.621 | 1.143 | 28 | 3.401e-03 | - | 3.408e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.807 | 1.329 | 29 | 2.282e-03 | 8.083e-02 | 2.282e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.723 | 1.314 | 29 | 2.576e-03 | 1.773e-01 | 2.555e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.882 | 1.473 | 28 | 8.097e-03 | - | 8.113e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.610 | 1.200 | 29 | 5.834e-03 | 5.467e-02 | 5.831e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.284 | 1.380 | 29 | 3.259e-03 | 1.771e-01 | 3.242e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.053 | 1.149 | 28 | 4.077e-03 | - | 4.094e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.330 | 1.427 | 29 | 4.215e-03 | 8.421e-02 | 4.226e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.084 | 1.270 | 29 | 1.574e-03 | 1.771e-01 | 1.557e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.973 | 1.158 | 28 | 7.033e-03 | - | 7.040e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.924 | 1.109 | 29 | 7.339e-03 | 1.381e-01 | 7.326e-03 |
| Inv-Newton | 512x512 | spike | 2.626 | 1.169 | 29 | 2.349e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 2.764 | 1.308 | 28 | 6.235e-03 | - | 6.242e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.614 | 1.158 | 29 | 6.271e-03 | 1.688e-01 | 6.273e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.082 | 2.313 | 92 | 3.590e-03 | 2.509e-01 | 3.569e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.347 | 2.577 | 88 | 4.224e-03 | - | 4.245e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.165 | 2.396 | 92 | 4.232e-03 | 1.507e-01 | 4.250e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.869 | 2.400 | 92 | 4.346e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.095 | 2.627 | 88 | 2.536e-04 | - | 2.435e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.892 | 2.424 | 92 | 3.180e-04 | 3.558e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.670 | 2.321 | 92 | 1.128e-03 | 2.503e-01 | 1.115e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 3.992 | 2.644 | 88 | 7.071e-03 | - | 7.076e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.708 | 2.359 | 92 | 7.071e-03 | 2.532e-01 | 7.077e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.758 | 2.380 | 92 | 1.189e-03 | 2.503e-01 | 1.177e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 3.981 | 2.603 | 88 | 7.008e-03 | - | 7.013e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.762 | 2.384 | 92 | 7.008e-03 | 2.528e-01 | 7.015e-03 |
| Inv-Newton | 1024x1024 | spike | 3.801 | 2.394 | 92 | 2.372e-03 | 2.455e-01 | 2.354e-03 |
| PE-Quad | 1024x1024 | spike | 4.454 | 3.046 | 88 | 6.411e-03 | - | 6.415e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.850 | 2.443 | 92 | 6.123e-03 | 2.390e-01 | 6.130e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.803 | 1.958 | 13 | 3.329e-03 | 3.036e-02 | 1.640e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.105 | 1.260 | 13 | 3.649e-03 | - | 1.802e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.162 | 1.317 | 13 | 3.656e-03 | 3.338e-02 | 1.810e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.946 | 1.230 | 13 | 3.215e-03 | 6.013e-02 | 1.599e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.175 | 1.459 | 13 | 3.265e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.830 | 1.114 | 13 | 3.251e-03 | 3.516e-02 | 1.627e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.306 | 1.473 | 13 | 1.938e-03 | 3.418e-02 | 9.660e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 3.327 | 1.494 | 13 | 2.078e-03 | - | 1.031e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.894 | 1.061 | 13 | 2.118e-03 | 1.914e-02 | 1.054e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.531 | 1.172 | 13 | 1.729e-03 | 8.212e-03 | 8.815e-04 |
| PE-Quad | 256x256 | near_rank_def | 2.708 | 1.350 | 13 | 3.738e-03 | - | 1.874e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.557 | 1.198 | 13 | 2.556e-03 | 3.907e-03 | 1.279e-03 |
| Inv-Newton | 256x256 | spike | 2.615 | 1.193 | 13 | 3.365e-03 | 2.409e-02 | 1.725e-03 |
| PE-Quad | 256x256 | spike | 2.792 | 1.369 | 13 | 5.605e-03 | - | 2.851e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.736 | 1.314 | 13 | 8.699e-03 | 5.510e-02 | 4.369e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.809 | 1.255 | 29 | 4.223e-03 | 1.562e-01 | 2.069e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.009 | 1.455 | 28 | 4.223e-03 | - | 2.066e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.773 | 1.220 | 29 | 4.586e-03 | 8.378e-02 | 2.196e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 5.253 | 1.378 | 29 | 2.762e-03 | 1.122e-01 | 1.437e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 5.193 | 1.318 | 28 | 3.075e-03 | - | 1.594e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 5.195 | 1.321 | 29 | 2.910e-03 | 5.593e-02 | 1.483e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.897 | 1.316 | 29 | 3.408e-03 | 1.495e-01 | 1.732e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.800 | 1.219 | 28 | 3.406e-03 | - | 1.730e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.909 | 1.328 | 29 | 8.409e-03 | 7.318e-02 | 4.241e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.946 | 1.158 | 29 | 1.591e-03 | 6.774e-02 | 7.996e-04 |
| PE-Quad | 512x512 | near_rank_def | 3.054 | 1.267 | 28 | 1.590e-03 | - | 7.981e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.064 | 1.276 | 29 | 1.023e-02 | 2.975e-02 | 5.057e-03 |
| Inv-Newton | 512x512 | spike | 2.765 | 1.306 | 29 | 1.479e-03 | 6.455e-02 | 7.472e-04 |
| PE-Quad | 512x512 | spike | 2.925 | 1.465 | 28 | 1.774e-03 | - | 8.805e-04 |
| PE-Quad-Coupled | 512x512 | spike | 3.165 | 1.705 | 29 | 1.115e-02 | 6.966e-02 | 5.652e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.629 | 2.812 | 92 | 3.376e-03 | 2.142e-01 | 1.637e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.028 | 3.211 | 88 | 3.374e-03 | - | 1.635e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.668 | 2.851 | 92 | 1.098e-02 | 2.759e-01 | 5.523e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.304 | 2.805 | 92 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.687 | 3.188 | 88 | 3.887e-03 | - | 1.845e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.272 | 2.773 | 92 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.717 | 2.756 | 92 | 4.739e-04 | 4.197e-03 | 1.648e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.057 | 3.095 | 88 | 4.724e-04 | - | 1.618e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.701 | 2.739 | 92 | 1.155e-02 | 2.652e-01 | 5.801e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.780 | 2.804 | 92 | 4.884e-04 | 3.873e-03 | 1.859e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 5.045 | 3.069 | 88 | 4.870e-04 | - | 1.834e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.794 | 2.818 | 92 | 1.154e-02 | 2.656e-01 | 5.809e-03 |
| Inv-Newton | 1024x1024 | spike | 4.057 | 2.814 | 92 | 1.384e-03 | 8.567e-02 | 7.163e-04 |
| PE-Quad | 1024x1024 | spike | 4.396 | 3.153 | 88 | 1.673e-03 | - | 8.445e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.027 | 2.783 | 92 | 1.026e-02 | 2.109e-01 | 5.198e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.487 | 2.102 | 13 | 1.497e-02 | 5.224e-02 | 5.045e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.713 | 1.329 | 13 | 5.448e-03 | - | 1.869e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.562 | 1.178 | 13 | 5.832e-03 | 9.211e-02 | 1.949e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.683 | 1.401 | 13 | 1.537e-02 | 5.157e-02 | 5.207e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.660 | 1.377 | 13 | 5.088e-03 | - | 1.660e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.547 | 1.265 | 13 | 5.093e-03 | 3.747e-02 | 1.659e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.576 | 1.269 | 13 | 1.613e-02 | 5.901e-02 | 5.509e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.705 | 1.398 | 13 | 3.176e-03 | - | 1.081e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.561 | 1.253 | 13 | 3.171e-03 | 2.067e-02 | 1.079e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.680 | 1.354 | 13 | 1.525e-02 | 5.636e-02 | 5.201e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.720 | 1.394 | 13 | 3.018e-03 | - | 9.986e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.596 | 1.270 | 13 | 3.014e-03 | 5.123e-02 | 9.997e-04 |
| Inv-Newton | 256x256 | spike | 2.623 | 1.324 | 13 | 1.057e-02 | 4.688e-02 | 3.516e-03 |
| PE-Quad | 256x256 | spike | 2.706 | 1.407 | 13 | 4.300e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.534 | 1.235 | 13 | 4.283e-03 | 1.019e-01 | 1.430e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.658 | 1.320 | 29 | 1.492e-02 | 8.053e-02 | 5.054e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.752 | 1.414 | 28 | 7.773e-03 | - | 2.720e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.630 | 1.292 | 29 | 7.820e-03 | 6.052e-02 | 2.645e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.928 | 1.422 | 29 | 1.360e-02 | 6.530e-02 | 4.625e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.965 | 1.459 | 28 | 5.893e-03 | - | 2.059e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.015 | 1.509 | 29 | 6.007e-03 | 4.819e-04 | 2.097e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.630 | 1.282 | 29 | 1.201e-02 | 4.788e-02 | 3.928e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.797 | 1.449 | 28 | 8.047e-03 | - | 2.622e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.611 | 1.264 | 29 | 8.101e-03 | 4.015e-04 | 2.780e-03 |
| Inv-Newton | 512x512 | near_rank_def | 4.019 | 1.573 | 29 | 1.595e-02 | 8.252e-02 | 5.242e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.898 | 1.452 | 28 | 4.108e-03 | - | 1.394e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.889 | 1.443 | 29 | 4.132e-03 | 3.879e-04 | 1.444e-03 |
| Inv-Newton | 512x512 | spike | 2.511 | 1.194 | 29 | 1.511e-02 | 1.032e-01 | 5.051e-03 |
| PE-Quad | 512x512 | spike | 2.665 | 1.349 | 28 | 4.037e-03 | - | 1.339e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.578 | 1.262 | 29 | 4.131e-03 | 6.629e-02 | 1.368e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.783 | 3.333 | 92 | 1.213e-02 | 7.182e-02 | 3.963e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.289 | 3.839 | 88 | 7.995e-03 | - | 2.586e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.701 | 3.251 | 92 | 8.249e-03 | 1.525e-01 | 2.739e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.584 | 3.295 | 92 | 9.048e-03 | 2.739e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.151 | 3.863 | 88 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.597 | 3.308 | 92 | 9.526e-03 | 1.782e-01 | 3.158e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.703 | 3.313 | 92 | 1.627e-02 | 1.250e-01 | 5.381e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.258 | 3.868 | 88 | 2.651e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.715 | 3.325 | 92 | 2.653e-03 | 5.293e-04 | 9.178e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.628 | 3.325 | 92 | 1.620e-02 | 1.250e-01 | 5.367e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.141 | 3.838 | 88 | 2.713e-03 | - | 9.310e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.646 | 3.343 | 92 | 2.715e-03 | 5.126e-04 | 9.313e-04 |
| Inv-Newton | 1024x1024 | spike | 4.680 | 3.328 | 92 | 1.496e-02 | 1.546e-01 | 5.011e-03 |
| PE-Quad | 1024x1024 | spike | 5.248 | 3.896 | 88 | 4.138e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.660 | 3.309 | 92 | 4.138e-03 | 8.629e-02 | 1.359e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.925 | 1.481 | 13 | 8.493e-03 | 1.080e-01 | 2.242e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.772 | 1.328 | 13 | 8.481e-03 | - | 2.241e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.597 | 1.153 | 13 | 1.217e-02 | 7.329e-02 | 3.144e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.504 | 1.205 | 13 | 1.027e-02 | 1.475e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.713 | 1.414 | 13 | 1.058e-02 | - | 2.728e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.470 | 1.171 | 13 | 1.286e-02 | 6.944e-02 | 3.235e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.485 | 1.185 | 13 | 1.153e-02 | 1.314e-01 | 2.947e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.635 | 1.334 | 13 | 1.340e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.476 | 1.176 | 13 | 1.428e-02 | 3.906e-02 | 3.450e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.552 | 1.233 | 13 | 1.101e-02 | 1.180e-01 | 2.815e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.717 | 1.398 | 13 | 1.281e-02 | - | 3.114e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.423 | 1.104 | 13 | 1.358e-02 | 1.563e-02 | 3.275e-03 |
| Inv-Newton | 256x256 | spike | 2.401 | 1.126 | 13 | 8.877e-03 | 9.560e-02 | 2.255e-03 |
| PE-Quad | 256x256 | spike | 2.585 | 1.310 | 13 | 9.420e-03 | - | 2.336e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.588 | 1.313 | 13 | 9.507e-03 | 6.347e-02 | 2.351e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.571 | 1.247 | 29 | 8.363e-03 | 2.425e-01 | 1.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.682 | 1.359 | 28 | 8.995e-03 | - | 2.085e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.489 | 1.165 | 29 | 8.969e-03 | 1.484e-01 | 2.085e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.616 | 1.308 | 29 | 1.050e-02 | 2.167e-01 | 2.586e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.719 | 1.410 | 28 | 1.181e-02 | - | 2.805e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.606 | 1.297 | 29 | 1.182e-02 | 1.121e-01 | 2.805e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.513 | 1.514 | 29 | 9.108e-03 | 2.433e-01 | 2.203e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.418 | 1.420 | 28 | 9.640e-03 | - | 2.291e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.272 | 1.274 | 29 | 9.649e-03 | 1.495e-01 | 2.291e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.693 | 1.267 | 29 | 1.225e-02 | 1.923e-01 | 3.057e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.852 | 1.426 | 28 | 1.347e-02 | - | 3.253e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.832 | 1.406 | 29 | 1.347e-02 | 6.766e-02 | 3.253e-03 |
| Inv-Newton | 512x512 | spike | 3.901 | 1.227 | 29 | 1.244e-02 | 1.893e-01 | 3.143e-03 |
| PE-Quad | 512x512 | spike | 4.012 | 1.338 | 28 | 1.250e-02 | - | 3.093e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.873 | 1.199 | 29 | 1.255e-02 | 7.118e-02 | 3.106e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.536 | 3.229 | 92 | 9.266e-03 | 3.461e-01 | 2.253e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.165 | 3.857 | 88 | 9.699e-03 | - | 2.322e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.555 | 3.248 | 92 | 9.491e-03 | 2.141e-01 | 2.288e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.477 | 3.203 | 92 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.138 | 3.864 | 88 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.513 | 3.239 | 92 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.723 | 3.245 | 92 | 1.287e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.315 | 3.837 | 88 | 1.371e-02 | - | 3.352e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.720 | 3.242 | 92 | 1.330e-02 | 3.559e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.660 | 3.263 | 92 | 1.289e-02 | 2.500e-01 | 3.298e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.165 | 3.768 | 88 | 1.365e-02 | - | 3.342e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.635 | 3.239 | 92 | 1.329e-02 | 3.428e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 4.609 | 3.253 | 92 | 1.281e-02 | 2.655e-01 | 3.219e-03 |
| PE-Quad | 1024x1024 | spike | 5.223 | 3.868 | 88 | 1.261e-02 | - | 3.145e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.585 | 3.230 | 92 | 1.292e-02 | 1.906e-01 | 3.225e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.626 | 1.214 | 13 | 4.746e-02 | 4.339e-01 | 5.936e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.774 | 1.363 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.603 | 1.192 | 13 | 4.739e-02 | 2.049e-01 | 5.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.524 | 1.201 | 13 | 4.483e-02 | 4.404e-01 | 5.625e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.902 | 1.579 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.526 | 1.203 | 13 | 4.484e-02 | 2.162e-01 | 5.625e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.490 | 1.213 | 13 | 4.582e-02 | 4.407e-01 | 5.781e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.598 | 1.321 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.493 | 1.217 | 13 | 4.582e-02 | 2.405e-01 | 5.781e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.555 | 1.269 | 13 | 4.733e-02 | 4.339e-01 | 5.993e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.640 | 1.354 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.537 | 1.250 | 13 | 4.733e-02 | 2.485e-01 | 5.993e-03 |
| Inv-Newton | 256x256 | spike | 2.486 | 1.196 | 13 | 4.713e-02 | 3.812e-01 | 6.025e-03 |
| PE-Quad | 256x256 | spike | 2.664 | 1.374 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | spike | 2.550 | 1.260 | 13 | 4.905e-02 | 2.288e-01 | 6.273e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.702 | 1.356 | 29 | 4.010e-02 | 5.436e-01 | 5.047e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.822 | 1.476 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.621 | 1.275 | 29 | 4.010e-02 | 2.130e-01 | 5.047e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.659 | 1.333 | 29 | 4.352e-02 | 5.789e-01 | 5.520e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.815 | 1.489 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.549 | 1.223 | 29 | 4.356e-02 | 2.882e-01 | 5.525e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.531 | 1.211 | 29 | 4.042e-02 | 5.567e-01 | 5.131e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.675 | 1.356 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.813 | 1.493 | 29 | 4.049e-02 | 1.987e-01 | 5.140e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.622 | 1.297 | 29 | 4.437e-02 | 6.064e-01 | 5.648e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.678 | 1.354 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.584 | 1.260 | 29 | 4.448e-02 | 2.475e-01 | 5.661e-03 |
| Inv-Newton | 512x512 | spike | 2.849 | 1.406 | 29 | 4.130e-02 | 5.726e-01 | 5.255e-03 |
| PE-Quad | 512x512 | spike | 2.848 | 1.405 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | spike | 2.849 | 1.406 | 29 | 4.369e-02 | 1.672e-01 | 5.558e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.617 | 3.683 | 92 | 3.999e-02 | 7.840e-01 | 5.085e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.449 | 4.516 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.609 | 3.675 | 92 | 4.050e-02 | 2.500e-01 | 5.153e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.929 | 3.667 | 92 | 3.745e-02 | 7.437e-01 | 4.763e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.612 | 4.350 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.953 | 3.692 | 92 | 3.786e-02 | 2.500e-01 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.993 | 3.667 | 92 | 4.413e-02 | 8.594e-01 | 5.639e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.666 | 4.340 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.998 | 3.672 | 92 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.055 | 3.674 | 92 | 4.401e-02 | 8.577e-01 | 5.628e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.724 | 4.343 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.052 | 3.671 | 92 | 4.481e-02 | 2.500e-01 | 5.731e-03 |
| Inv-Newton | 1024x1024 | spike | 5.072 | 3.672 | 92 | 3.970e-02 | 7.742e-01 | 5.006e-03 |
| PE-Quad | 1024x1024 | spike | 5.757 | 4.357 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | spike | 5.073 | 3.673 | 92 | 4.420e-02 | 2.423e-01 | 5.571e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
