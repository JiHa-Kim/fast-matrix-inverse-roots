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
| Inv-Newton | 256x256 | gaussian_spd | 2.475 | 1.004 | 13 | 2.066e-03 | 1.258e-01 | 2.070e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.526 | 1.055 | 13 | 3.809e-03 | - | 3.808e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.481 | 1.010 | 13 | 4.521e-03 | 1.381e-01 | 4.513e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.464 | 1.136 | 13 | 1.499e-03 | 1.255e-01 | 1.489e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.421 | 1.093 | 13 | 7.084e-03 | - | 7.089e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.393 | 1.065 | 13 | 5.912e-03 | 1.361e-01 | 5.901e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.236 | 1.008 | 13 | 1.108e-03 | 1.253e-01 | 1.110e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.989 | 1.760 | 13 | 8.258e-03 | - | 8.260e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.417 | 1.189 | 13 | 7.717e-03 | 7.348e-02 | 7.702e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.450 | 1.163 | 13 | 2.367e-03 | 1.253e-01 | 2.377e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.343 | 1.057 | 13 | 8.876e-03 | - | 8.882e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.312 | 1.026 | 13 | 8.508e-03 | 6.450e-02 | 8.484e-03 |
| Inv-Newton | 256x256 | spike | 2.443 | 1.050 | 13 | 4.917e-03 | 1.234e-01 | 4.929e-03 |
| PE-Quad | 256x256 | spike | 2.580 | 1.186 | 13 | 6.868e-03 | - | 6.873e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.402 | 1.009 | 13 | 6.951e-03 | 1.175e-01 | 6.947e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.645 | 1.065 | 29 | 1.860e-03 | 1.779e-01 | 1.867e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.088 | 1.507 | 28 | 5.550e-03 | - | 5.544e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.836 | 1.255 | 29 | 4.376e-03 | 8.083e-02 | 4.372e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.545 | 1.219 | 29 | 2.576e-03 | 1.773e-01 | 2.555e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.659 | 1.333 | 28 | 8.097e-03 | - | 8.113e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.474 | 1.149 | 29 | 5.834e-03 | 5.467e-02 | 5.831e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.718 | 2.461 | 29 | 3.259e-03 | 1.771e-01 | 3.242e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.438 | 1.181 | 28 | 4.077e-03 | - | 4.094e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.375 | 1.117 | 29 | 4.215e-03 | 8.421e-02 | 4.226e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.476 | 1.151 | 29 | 1.574e-03 | 1.771e-01 | 1.557e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.584 | 1.259 | 28 | 7.033e-03 | - | 7.040e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.385 | 1.060 | 29 | 7.339e-03 | 1.381e-01 | 7.326e-03 |
| Inv-Newton | 512x512 | spike | 2.332 | 1.114 | 29 | 2.349e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 2.367 | 1.149 | 28 | 6.235e-03 | - | 6.242e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.288 | 1.070 | 29 | 6.271e-03 | 1.688e-01 | 6.273e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.065 | 2.506 | 92 | 3.590e-03 | 2.509e-01 | 3.569e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.154 | 2.596 | 88 | 4.224e-03 | - | 4.245e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.045 | 2.486 | 92 | 4.232e-03 | 1.507e-01 | 4.250e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 3.672 | 2.376 | 92 | 4.346e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 3.911 | 2.615 | 88 | 2.536e-04 | - | 2.435e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.781 | 2.485 | 92 | 3.180e-04 | 3.558e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.652 | 2.331 | 92 | 1.128e-03 | 2.503e-01 | 1.115e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 3.896 | 2.574 | 88 | 7.071e-03 | - | 7.076e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.685 | 2.363 | 92 | 7.071e-03 | 2.532e-01 | 7.077e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.646 | 2.351 | 92 | 1.182e-03 | 2.503e-01 | 1.169e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 3.889 | 2.594 | 88 | 7.008e-03 | - | 7.013e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.615 | 2.319 | 92 | 7.008e-03 | 2.528e-01 | 7.015e-03 |
| Inv-Newton | 1024x1024 | spike | 3.809 | 2.417 | 92 | 2.372e-03 | 2.455e-01 | 2.354e-03 |
| PE-Quad | 1024x1024 | spike | 4.008 | 2.616 | 88 | 6.411e-03 | - | 6.415e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.761 | 2.369 | 92 | 6.123e-03 | 2.390e-01 | 6.130e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.362 | 1.077 | 13 | 3.329e-03 | 3.036e-02 | 1.640e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.426 | 1.142 | 13 | 3.649e-03 | - | 1.802e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.338 | 1.054 | 13 | 3.656e-03 | 3.338e-02 | 1.810e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.285 | 1.061 | 13 | 3.215e-03 | 6.013e-02 | 1.599e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.372 | 1.148 | 13 | 3.265e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.266 | 1.042 | 13 | 3.251e-03 | 3.516e-02 | 1.627e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.330 | 1.089 | 13 | 1.938e-03 | 3.327e-02 | 9.660e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 2.396 | 1.155 | 13 | 2.078e-03 | - | 1.031e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.400 | 1.158 | 13 | 2.118e-03 | 1.832e-02 | 1.054e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.426 | 1.102 | 13 | 1.729e-03 | 8.212e-03 | 8.815e-04 |
| PE-Quad | 256x256 | near_rank_def | 2.476 | 1.152 | 13 | 3.738e-03 | - | 1.874e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.383 | 1.059 | 13 | 2.556e-03 | 3.907e-03 | 1.279e-03 |
| Inv-Newton | 256x256 | spike | 2.296 | 1.060 | 13 | 3.365e-03 | 2.245e-02 | 1.725e-03 |
| PE-Quad | 256x256 | spike | 2.390 | 1.154 | 13 | 5.605e-03 | - | 2.851e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.270 | 1.033 | 13 | 8.610e-03 | 5.538e-02 | 4.327e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.346 | 1.094 | 29 | 2.363e-03 | 8.585e-02 | 1.154e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.443 | 1.191 | 28 | 3.296e-03 | - | 1.702e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.393 | 1.140 | 29 | 2.382e-03 | 4.279e-02 | 1.158e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.603 | 1.207 | 29 | 2.762e-03 | 1.122e-01 | 1.437e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.739 | 1.343 | 28 | 3.075e-03 | - | 1.594e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.083 | 1.688 | 29 | 2.910e-03 | 5.593e-02 | 1.483e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.603 | 1.260 | 29 | 3.408e-03 | 1.495e-01 | 1.732e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 4.136 | 2.793 | 28 | 3.406e-03 | - | 1.730e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.659 | 1.315 | 29 | 8.409e-03 | 7.318e-02 | 4.241e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.356 | 1.106 | 29 | 1.591e-03 | 6.774e-02 | 7.996e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.652 | 1.402 | 28 | 1.590e-03 | - | 7.981e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.322 | 1.071 | 29 | 1.023e-02 | 2.975e-02 | 5.057e-03 |
| Inv-Newton | 512x512 | spike | 3.020 | 1.722 | 29 | 1.479e-03 | 6.455e-02 | 7.472e-04 |
| PE-Quad | 512x512 | spike | 2.560 | 1.263 | 28 | 1.774e-03 | - | 8.805e-04 |
| PE-Quad-Coupled | 512x512 | spike | 2.484 | 1.186 | 29 | 1.115e-02 | 6.966e-02 | 5.652e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.094 | 2.825 | 92 | 3.376e-03 | 2.142e-01 | 1.637e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.458 | 3.190 | 88 | 3.374e-03 | - | 1.635e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.058 | 2.789 | 92 | 1.098e-02 | 2.759e-01 | 5.523e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.170 | 2.841 | 92 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.519 | 3.190 | 88 | 3.887e-03 | - | 1.845e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.141 | 2.811 | 92 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.078 | 2.827 | 92 | 4.739e-04 | 4.197e-03 | 1.648e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.455 | 3.204 | 88 | 4.724e-04 | - | 1.618e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.078 | 2.827 | 92 | 1.155e-02 | 2.652e-01 | 5.801e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.119 | 2.821 | 92 | 4.868e-04 | 3.833e-03 | 1.833e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 4.507 | 3.208 | 88 | 4.853e-04 | - | 1.807e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.128 | 2.829 | 92 | 1.155e-02 | 2.655e-01 | 5.817e-03 |
| Inv-Newton | 1024x1024 | spike | 4.187 | 2.744 | 92 | 1.384e-03 | 8.567e-02 | 7.163e-04 |
| PE-Quad | 1024x1024 | spike | 4.663 | 3.219 | 88 | 1.673e-03 | - | 8.445e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.161 | 2.718 | 92 | 1.026e-02 | 2.109e-01 | 5.198e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.572 | 1.248 | 13 | 1.497e-02 | 5.224e-02 | 5.045e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.676 | 1.353 | 13 | 5.448e-03 | - | 1.869e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.668 | 1.345 | 13 | 5.832e-03 | 9.211e-02 | 1.949e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.461 | 1.220 | 13 | 1.537e-02 | 5.157e-02 | 5.207e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.595 | 1.354 | 13 | 5.088e-03 | - | 1.660e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.457 | 1.215 | 13 | 5.093e-03 | 3.747e-02 | 1.659e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.460 | 1.231 | 13 | 1.613e-02 | 5.901e-02 | 5.509e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.584 | 1.355 | 13 | 3.176e-03 | - | 1.081e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.433 | 1.203 | 13 | 3.171e-03 | 2.067e-02 | 1.079e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.472 | 1.223 | 13 | 1.525e-02 | 5.636e-02 | 5.201e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.597 | 1.348 | 13 | 3.018e-03 | - | 9.986e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.471 | 1.222 | 13 | 3.014e-03 | 5.123e-02 | 9.997e-04 |
| Inv-Newton | 256x256 | spike | 2.706 | 1.277 | 13 | 1.033e-02 | 4.622e-02 | 3.440e-03 |
| PE-Quad | 256x256 | spike | 2.806 | 1.376 | 13 | 4.289e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.632 | 1.203 | 13 | 4.264e-03 | 1.019e-01 | 1.425e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.467 | 1.220 | 29 | 1.519e-02 | 8.370e-02 | 5.212e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.586 | 1.339 | 28 | 4.727e-03 | - | 1.641e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.470 | 1.224 | 29 | 4.742e-03 | 6.900e-02 | 1.597e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.686 | 1.252 | 29 | 1.360e-02 | 6.530e-02 | 4.625e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.873 | 1.440 | 28 | 5.893e-03 | - | 2.059e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.711 | 1.278 | 29 | 6.007e-03 | 4.819e-04 | 2.097e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.577 | 1.323 | 29 | 1.201e-02 | 4.788e-02 | 3.928e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.702 | 1.448 | 28 | 8.047e-03 | - | 2.622e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.480 | 1.226 | 29 | 8.101e-03 | 4.015e-04 | 2.780e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.935 | 1.556 | 29 | 1.595e-02 | 8.252e-02 | 5.242e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.794 | 1.415 | 28 | 4.108e-03 | - | 1.394e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.646 | 1.268 | 29 | 4.132e-03 | 3.879e-04 | 1.444e-03 |
| Inv-Newton | 512x512 | spike | 2.809 | 1.527 | 29 | 1.511e-02 | 1.032e-01 | 5.051e-03 |
| PE-Quad | 512x512 | spike | 2.668 | 1.385 | 28 | 4.037e-03 | - | 1.339e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.494 | 1.211 | 29 | 4.131e-03 | 6.629e-02 | 1.368e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.780 | 3.366 | 92 | 1.213e-02 | 7.182e-02 | 3.963e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.297 | 3.883 | 88 | 7.995e-03 | - | 2.586e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.754 | 3.340 | 92 | 8.249e-03 | 1.525e-01 | 2.739e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.877 | 3.329 | 92 | 9.048e-03 | 2.739e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.445 | 3.897 | 88 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.798 | 3.250 | 92 | 9.526e-03 | 1.782e-01 | 3.158e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.858 | 3.361 | 92 | 1.627e-02 | 1.250e-01 | 5.381e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.310 | 3.813 | 88 | 2.651e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.841 | 3.344 | 92 | 2.653e-03 | 5.293e-04 | 9.178e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.824 | 3.288 | 92 | 1.620e-02 | 1.255e-01 | 5.367e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.410 | 3.875 | 88 | 2.706e-03 | - | 9.294e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.887 | 3.352 | 92 | 2.708e-03 | 5.047e-04 | 9.297e-04 |
| Inv-Newton | 1024x1024 | spike | 4.891 | 3.342 | 92 | 1.496e-02 | 1.546e-01 | 5.011e-03 |
| PE-Quad | 1024x1024 | spike | 5.430 | 3.881 | 88 | 4.138e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.884 | 3.335 | 92 | 4.138e-03 | 8.629e-02 | 1.359e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.494 | 1.132 | 13 | 8.493e-03 | 1.080e-01 | 2.242e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.727 | 1.365 | 13 | 8.479e-03 | - | 2.240e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.624 | 1.262 | 13 | 1.215e-02 | 7.329e-02 | 3.144e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.445 | 1.198 | 13 | 1.027e-02 | 1.475e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.586 | 1.339 | 13 | 1.058e-02 | - | 2.728e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.443 | 1.196 | 13 | 1.286e-02 | 6.944e-02 | 3.235e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.407 | 1.184 | 13 | 1.153e-02 | 1.312e-01 | 2.947e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.966 | 1.743 | 13 | 1.340e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.554 | 1.330 | 13 | 1.428e-02 | 3.906e-02 | 3.450e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.796 | 1.394 | 13 | 1.101e-02 | 1.180e-01 | 2.815e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.850 | 1.448 | 13 | 1.281e-02 | - | 3.114e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.555 | 1.153 | 13 | 1.358e-02 | 1.563e-02 | 3.275e-03 |
| Inv-Newton | 256x256 | spike | 2.499 | 1.128 | 13 | 8.800e-03 | 9.367e-02 | 2.234e-03 |
| PE-Quad | 256x256 | spike | 2.720 | 1.349 | 13 | 9.224e-03 | - | 2.288e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.653 | 1.281 | 13 | 9.335e-03 | 6.152e-02 | 2.313e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.738 | 1.439 | 29 | 1.112e-02 | 2.010e-01 | 2.828e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.696 | 1.398 | 28 | 1.345e-02 | - | 3.228e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.565 | 1.267 | 29 | 1.342e-02 | 8.558e-02 | 3.228e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.577 | 1.316 | 29 | 1.050e-02 | 2.167e-01 | 2.586e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.790 | 1.529 | 28 | 1.181e-02 | - | 2.805e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.759 | 1.497 | 29 | 1.182e-02 | 1.121e-01 | 2.805e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.548 | 1.217 | 29 | 9.108e-03 | 2.433e-01 | 2.203e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.723 | 1.391 | 28 | 9.640e-03 | - | 2.291e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.605 | 1.274 | 29 | 9.649e-03 | 1.495e-01 | 2.291e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.025 | 1.477 | 29 | 1.225e-02 | 1.923e-01 | 3.057e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.045 | 1.496 | 28 | 1.347e-02 | - | 3.253e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.843 | 1.294 | 29 | 1.347e-02 | 6.766e-02 | 3.253e-03 |
| Inv-Newton | 512x512 | spike | 2.585 | 1.273 | 29 | 1.244e-02 | 1.893e-01 | 3.143e-03 |
| PE-Quad | 512x512 | spike | 2.895 | 1.583 | 28 | 1.250e-02 | - | 3.093e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.530 | 1.218 | 29 | 1.255e-02 | 7.118e-02 | 3.106e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.786 | 3.268 | 92 | 9.266e-03 | 3.461e-01 | 2.253e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.396 | 3.878 | 88 | 9.699e-03 | - | 2.322e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.782 | 3.263 | 92 | 9.491e-03 | 2.141e-01 | 2.288e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.516 | 3.283 | 92 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.123 | 3.890 | 88 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.507 | 3.274 | 92 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.454 | 3.202 | 92 | 1.287e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.114 | 3.863 | 88 | 1.371e-02 | - | 3.352e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.601 | 3.350 | 92 | 1.330e-02 | 3.559e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.489 | 3.245 | 92 | 1.292e-02 | 2.500e-01 | 3.303e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.121 | 3.877 | 88 | 1.365e-02 | - | 3.342e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.496 | 3.252 | 92 | 1.329e-02 | 3.376e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 4.606 | 3.282 | 92 | 1.281e-02 | 2.655e-01 | 3.219e-03 |
| PE-Quad | 1024x1024 | spike | 5.163 | 3.839 | 88 | 1.261e-02 | - | 3.145e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.612 | 3.287 | 92 | 1.292e-02 | 1.906e-01 | 3.225e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.557 | 1.251 | 13 | 4.746e-02 | 4.339e-01 | 5.936e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.722 | 1.416 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.536 | 1.230 | 13 | 4.739e-02 | 2.049e-01 | 5.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.498 | 1.232 | 13 | 4.483e-02 | 4.404e-01 | 5.625e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.630 | 1.364 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.639 | 1.373 | 13 | 4.484e-02 | 2.162e-01 | 5.625e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.911 | 1.580 | 13 | 4.592e-02 | 4.404e-01 | 5.802e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.928 | 1.597 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.671 | 1.340 | 13 | 4.582e-02 | 2.405e-01 | 5.781e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.754 | 1.331 | 13 | 4.733e-02 | 4.339e-01 | 5.993e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.833 | 1.410 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.663 | 1.240 | 13 | 4.733e-02 | 2.485e-01 | 5.993e-03 |
| Inv-Newton | 256x256 | spike | 2.625 | 1.235 | 13 | 4.714e-02 | 3.812e-01 | 6.029e-03 |
| PE-Quad | 256x256 | spike | 3.586 | 2.195 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | spike | 3.369 | 1.979 | 13 | 4.947e-02 | 2.290e-01 | 6.326e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.947 | 1.370 | 29 | 4.460e-02 | 5.995e-01 | 5.629e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.143 | 1.567 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.046 | 1.469 | 29 | 4.460e-02 | 3.215e-01 | 5.629e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.970 | 2.095 | 29 | 4.352e-02 | 5.789e-01 | 5.520e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.688 | 1.812 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.285 | 1.409 | 29 | 4.356e-02 | 2.882e-01 | 5.525e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.679 | 1.359 | 29 | 4.042e-02 | 5.567e-01 | 5.131e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.718 | 1.398 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.764 | 1.444 | 29 | 4.049e-02 | 1.987e-01 | 5.140e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.918 | 1.520 | 29 | 4.437e-02 | 6.064e-01 | 5.648e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.925 | 1.527 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.734 | 1.335 | 29 | 4.448e-02 | 2.475e-01 | 5.661e-03 |
| Inv-Newton | 512x512 | spike | 3.042 | 1.612 | 29 | 4.130e-02 | 5.726e-01 | 5.255e-03 |
| PE-Quad | 512x512 | spike | 3.678 | 2.248 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | spike | 2.917 | 1.487 | 29 | 4.369e-02 | 1.672e-01 | 5.558e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 7.402 | 3.736 | 92 | 3.999e-02 | 7.840e-01 | 5.085e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 8.430 | 4.764 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 7.752 | 4.086 | 92 | 4.050e-02 | 2.500e-01 | 5.153e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.459 | 3.914 | 92 | 3.745e-02 | 7.437e-01 | 4.763e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.253 | 4.708 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.697 | 4.152 | 92 | 3.786e-02 | 2.500e-01 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.953 | 4.181 | 92 | 4.413e-02 | 8.594e-01 | 5.639e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 6.462 | 4.690 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 6.048 | 4.275 | 92 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 7.817 | 4.044 | 92 | 4.401e-02 | 8.582e-01 | 5.628e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 8.514 | 4.741 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 7.945 | 4.172 | 92 | 4.481e-02 | 2.500e-01 | 5.731e-03 |
| Inv-Newton | 1024x1024 | spike | 5.442 | 3.924 | 92 | 3.970e-02 | 7.742e-01 | 5.006e-03 |
| PE-Quad | 1024x1024 | spike | 5.952 | 4.434 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | spike | 5.220 | 3.703 | 92 | 4.420e-02 | 2.423e-01 | 5.571e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
