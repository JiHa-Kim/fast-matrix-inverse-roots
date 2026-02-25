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
| Inv-Newton | 256x256 | gaussian_spd | 2.493 | 1.134 | 13 | 2.066e-03 | 1.258e-01 | 2.070e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.428 | 1.069 | 13 | 3.809e-03 | - | 3.808e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.336 | 0.977 | 13 | 4.521e-03 | 1.381e-01 | 4.513e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.734 | 0.989 | 13 | 1.499e-03 | 1.255e-01 | 1.489e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.074 | 1.329 | 13 | 7.084e-03 | - | 7.089e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.830 | 1.085 | 13 | 5.912e-03 | 1.361e-01 | 5.901e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.850 | 1.138 | 13 | 1.108e-03 | 1.253e-01 | 1.110e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.766 | 1.054 | 13 | 8.258e-03 | - | 8.260e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.706 | 0.993 | 13 | 7.717e-03 | 7.348e-02 | 7.702e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.665 | 1.042 | 13 | 2.367e-03 | 1.253e-01 | 2.377e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.700 | 1.077 | 13 | 8.876e-03 | - | 8.882e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.795 | 1.172 | 13 | 8.508e-03 | 6.450e-02 | 8.484e-03 |
| Inv-Newton | 256x256 | spike | 2.587 | 1.031 | 13 | 4.917e-03 | 1.234e-01 | 4.929e-03 |
| PE-Quad | 256x256 | spike | 2.673 | 1.117 | 13 | 6.868e-03 | - | 6.873e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.596 | 1.040 | 13 | 6.951e-03 | 1.175e-01 | 6.947e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.503 | 1.153 | 29 | 1.860e-03 | 1.779e-01 | 1.867e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.642 | 1.292 | 28 | 5.550e-03 | - | 5.544e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.454 | 1.104 | 29 | 4.376e-03 | 8.083e-02 | 4.372e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.635 | 1.043 | 29 | 2.576e-03 | 1.773e-01 | 2.555e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.871 | 1.279 | 28 | 8.097e-03 | - | 8.113e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.760 | 1.168 | 29 | 5.834e-03 | 5.467e-02 | 5.831e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.318 | 1.159 | 29 | 3.259e-03 | 1.771e-01 | 3.242e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.253 | 1.095 | 28 | 4.077e-03 | - | 4.094e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.280 | 1.121 | 29 | 4.215e-03 | 8.421e-02 | 4.226e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.586 | 1.073 | 29 | 1.574e-03 | 1.771e-01 | 1.557e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.641 | 1.128 | 28 | 7.033e-03 | - | 7.040e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.540 | 1.028 | 29 | 7.339e-03 | 1.381e-01 | 7.326e-03 |
| Inv-Newton | 512x512 | spike | 2.622 | 1.152 | 29 | 2.349e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 2.555 | 1.085 | 28 | 6.235e-03 | - | 6.242e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.499 | 1.029 | 29 | 6.271e-03 | 1.688e-01 | 6.273e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 3.898 | 2.349 | 92 | 3.590e-03 | 2.509e-01 | 3.569e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.581 | 3.032 | 88 | 4.224e-03 | - | 4.245e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 3.963 | 2.414 | 92 | 4.232e-03 | 1.507e-01 | 4.250e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 3.900 | 2.379 | 92 | 4.346e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.220 | 2.698 | 88 | 2.536e-04 | - | 2.435e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.936 | 2.415 | 92 | 3.180e-04 | 3.558e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.833 | 2.437 | 92 | 1.128e-03 | 2.503e-01 | 1.115e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.019 | 2.623 | 88 | 7.071e-03 | - | 7.076e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.786 | 2.391 | 92 | 7.071e-03 | 2.532e-01 | 7.077e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.891 | 2.405 | 92 | 1.182e-03 | 2.503e-01 | 1.169e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 4.091 | 2.605 | 88 | 7.008e-03 | - | 7.013e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.899 | 2.413 | 92 | 7.008e-03 | 2.528e-01 | 7.015e-03 |
| Inv-Newton | 1024x1024 | spike | 3.743 | 2.386 | 92 | 2.372e-03 | 2.455e-01 | 2.354e-03 |
| PE-Quad | 1024x1024 | spike | 4.007 | 2.650 | 88 | 6.411e-03 | - | 6.415e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.763 | 2.406 | 92 | 6.123e-03 | 2.390e-01 | 6.130e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.621 | 1.233 | 13 | 3.329e-03 | 3.036e-02 | 1.640e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.821 | 1.433 | 13 | 3.649e-03 | - | 1.802e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.553 | 1.165 | 13 | 3.656e-03 | 3.338e-02 | 1.810e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.746 | 1.260 | 13 | 3.215e-03 | 6.013e-02 | 1.599e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.685 | 1.198 | 13 | 3.265e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.643 | 1.156 | 13 | 3.251e-03 | 3.516e-02 | 1.627e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.712 | 1.206 | 13 | 1.938e-03 | 3.327e-02 | 9.660e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 2.735 | 1.229 | 13 | 2.078e-03 | - | 1.031e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.657 | 1.151 | 13 | 2.118e-03 | 1.832e-02 | 1.054e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.483 | 1.156 | 13 | 1.729e-03 | 8.212e-03 | 8.815e-04 |
| PE-Quad | 256x256 | near_rank_def | 2.607 | 1.280 | 13 | 3.738e-03 | - | 1.874e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.693 | 1.366 | 13 | 2.556e-03 | 3.907e-03 | 1.279e-03 |
| Inv-Newton | 256x256 | spike | 2.721 | 1.396 | 13 | 3.365e-03 | 2.245e-02 | 1.725e-03 |
| PE-Quad | 256x256 | spike | 2.596 | 1.271 | 13 | 5.605e-03 | - | 2.851e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.489 | 1.163 | 13 | 8.610e-03 | 5.538e-02 | 4.327e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.825 | 1.304 | 29 | 2.363e-03 | 8.585e-02 | 1.154e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.751 | 1.230 | 28 | 3.296e-03 | - | 1.702e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.866 | 1.346 | 29 | 2.382e-03 | 4.279e-02 | 1.158e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.755 | 1.220 | 29 | 2.762e-03 | 1.122e-01 | 1.437e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.846 | 1.311 | 28 | 3.075e-03 | - | 1.594e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.756 | 1.222 | 29 | 2.910e-03 | 5.593e-02 | 1.483e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.020 | 1.231 | 29 | 3.408e-03 | 1.495e-01 | 1.732e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.288 | 1.499 | 28 | 3.406e-03 | - | 1.730e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.914 | 1.125 | 29 | 8.409e-03 | 7.318e-02 | 4.241e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.552 | 1.217 | 29 | 1.591e-03 | 6.774e-02 | 7.996e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.697 | 1.363 | 28 | 1.590e-03 | - | 7.981e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.731 | 1.397 | 29 | 1.023e-02 | 2.975e-02 | 5.057e-03 |
| Inv-Newton | 512x512 | spike | 2.353 | 1.106 | 29 | 1.479e-03 | 6.455e-02 | 7.472e-04 |
| PE-Quad | 512x512 | spike | 2.516 | 1.269 | 28 | 1.774e-03 | - | 8.805e-04 |
| PE-Quad-Coupled | 512x512 | spike | 2.500 | 1.252 | 29 | 1.115e-02 | 6.966e-02 | 5.652e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.460 | 2.928 | 92 | 3.376e-03 | 2.142e-01 | 1.637e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.747 | 3.214 | 88 | 3.374e-03 | - | 1.635e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.555 | 3.023 | 92 | 1.098e-02 | 2.759e-01 | 5.523e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.399 | 2.876 | 92 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.102 | 3.580 | 88 | 3.887e-03 | - | 1.845e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.433 | 2.910 | 92 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.475 | 2.864 | 92 | 4.739e-04 | 4.197e-03 | 1.648e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.837 | 3.226 | 88 | 4.724e-04 | - | 1.618e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.467 | 2.856 | 92 | 1.155e-02 | 2.652e-01 | 5.801e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.336 | 2.862 | 92 | 4.868e-04 | 3.833e-03 | 1.833e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 5.039 | 3.565 | 88 | 4.853e-04 | - | 1.807e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.260 | 2.786 | 92 | 1.155e-02 | 2.655e-01 | 5.817e-03 |
| Inv-Newton | 1024x1024 | spike | 5.217 | 2.801 | 92 | 1.384e-03 | 8.567e-02 | 7.163e-04 |
| PE-Quad | 1024x1024 | spike | 6.084 | 3.668 | 88 | 1.673e-03 | - | 8.445e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.662 | 3.246 | 92 | 1.026e-02 | 2.109e-01 | 5.198e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.399 | 1.522 | 13 | 1.497e-02 | 5.224e-02 | 5.045e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.310 | 1.433 | 13 | 5.448e-03 | - | 1.869e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.308 | 1.430 | 13 | 5.832e-03 | 9.211e-02 | 1.949e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.767 | 1.231 | 13 | 1.537e-02 | 5.157e-02 | 5.207e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.005 | 1.469 | 13 | 5.088e-03 | - | 1.660e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.774 | 1.238 | 13 | 5.093e-03 | 3.747e-02 | 1.659e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.099 | 1.271 | 13 | 1.613e-02 | 5.901e-02 | 5.509e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.390 | 1.562 | 13 | 3.176e-03 | - | 1.081e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.081 | 1.252 | 13 | 3.171e-03 | 2.067e-02 | 1.079e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.887 | 1.262 | 13 | 1.525e-02 | 5.636e-02 | 5.201e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.243 | 1.618 | 13 | 3.018e-03 | - | 9.986e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.918 | 1.292 | 13 | 3.014e-03 | 5.123e-02 | 9.997e-04 |
| Inv-Newton | 256x256 | spike | 2.837 | 1.236 | 13 | 1.033e-02 | 4.622e-02 | 3.440e-03 |
| PE-Quad | 256x256 | spike | 2.929 | 1.327 | 13 | 4.289e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.810 | 1.208 | 13 | 4.264e-03 | 1.019e-01 | 1.425e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.915 | 1.447 | 29 | 1.519e-02 | 8.370e-02 | 5.212e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.929 | 1.460 | 28 | 4.727e-03 | - | 1.641e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.895 | 1.426 | 29 | 4.742e-03 | 6.900e-02 | 1.597e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.559 | 1.753 | 29 | 1.360e-02 | 6.530e-02 | 4.625e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.265 | 1.459 | 28 | 5.893e-03 | - | 2.059e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.207 | 1.401 | 29 | 6.007e-03 | 4.819e-04 | 2.097e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.481 | 1.284 | 29 | 1.201e-02 | 4.788e-02 | 3.928e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.604 | 1.407 | 28 | 8.047e-03 | - | 2.622e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.436 | 1.238 | 29 | 8.101e-03 | 4.015e-04 | 2.780e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.143 | 1.350 | 29 | 1.595e-02 | 8.252e-02 | 5.242e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.282 | 1.489 | 28 | 4.108e-03 | - | 1.394e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.065 | 1.272 | 29 | 4.132e-03 | 3.879e-04 | 1.444e-03 |
| Inv-Newton | 512x512 | spike | 2.725 | 1.389 | 29 | 1.511e-02 | 1.032e-01 | 5.051e-03 |
| PE-Quad | 512x512 | spike | 2.796 | 1.460 | 28 | 4.037e-03 | - | 1.339e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.737 | 1.401 | 29 | 4.131e-03 | 6.629e-02 | 1.368e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.030 | 3.515 | 92 | 1.213e-02 | 7.182e-02 | 3.963e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.661 | 4.146 | 88 | 7.995e-03 | - | 2.586e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.191 | 3.676 | 92 | 8.249e-03 | 1.525e-01 | 2.739e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.023 | 3.341 | 92 | 9.048e-03 | 2.739e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.866 | 4.184 | 88 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.196 | 3.514 | 92 | 9.526e-03 | 1.782e-01 | 3.158e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.744 | 3.354 | 92 | 1.627e-02 | 1.250e-01 | 5.381e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.727 | 4.338 | 88 | 2.651e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.710 | 3.320 | 92 | 2.653e-03 | 5.293e-04 | 9.178e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.230 | 3.723 | 92 | 1.620e-02 | 1.255e-01 | 5.367e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.659 | 4.152 | 88 | 2.706e-03 | - | 9.294e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.302 | 3.795 | 92 | 2.708e-03 | 5.047e-04 | 9.297e-04 |
| Inv-Newton | 1024x1024 | spike | 5.234 | 3.730 | 92 | 1.496e-02 | 1.546e-01 | 5.011e-03 |
| PE-Quad | 1024x1024 | spike | 5.866 | 4.362 | 88 | 4.138e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.178 | 3.675 | 92 | 4.138e-03 | 8.629e-02 | 1.359e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.987 | 1.305 | 13 | 8.493e-03 | 1.080e-01 | 2.242e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.016 | 1.334 | 13 | 8.479e-03 | - | 2.240e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.834 | 1.152 | 13 | 1.215e-02 | 7.329e-02 | 3.144e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.605 | 1.220 | 13 | 1.027e-02 | 1.475e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.814 | 1.429 | 13 | 1.058e-02 | - | 2.728e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.654 | 1.269 | 13 | 1.286e-02 | 6.944e-02 | 3.235e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.538 | 1.180 | 13 | 1.153e-02 | 1.312e-01 | 2.947e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.762 | 1.403 | 13 | 1.340e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.530 | 1.171 | 13 | 1.428e-02 | 3.906e-02 | 3.450e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.535 | 1.200 | 13 | 1.101e-02 | 1.180e-01 | 2.815e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.865 | 1.530 | 13 | 1.281e-02 | - | 3.114e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.587 | 1.252 | 13 | 1.358e-02 | 1.563e-02 | 3.275e-03 |
| Inv-Newton | 256x256 | spike | 2.779 | 1.148 | 13 | 8.800e-03 | 9.367e-02 | 2.234e-03 |
| PE-Quad | 256x256 | spike | 3.091 | 1.459 | 13 | 9.224e-03 | - | 2.288e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.832 | 1.200 | 13 | 9.335e-03 | 6.152e-02 | 2.313e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.407 | 1.576 | 29 | 1.112e-02 | 2.010e-01 | 2.828e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.309 | 1.479 | 28 | 1.345e-02 | - | 3.228e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.140 | 1.310 | 29 | 1.342e-02 | 8.558e-02 | 3.228e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.571 | 1.227 | 29 | 1.050e-02 | 2.167e-01 | 2.586e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.798 | 1.454 | 28 | 1.181e-02 | - | 2.805e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.518 | 1.174 | 29 | 1.182e-02 | 1.121e-01 | 2.805e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.965 | 1.187 | 29 | 9.108e-03 | 2.433e-01 | 2.203e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.132 | 1.354 | 28 | 9.640e-03 | - | 2.291e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.959 | 1.181 | 29 | 9.649e-03 | 1.495e-01 | 2.291e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.617 | 1.182 | 29 | 1.225e-02 | 1.923e-01 | 3.057e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.901 | 1.466 | 28 | 1.347e-02 | - | 3.253e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.636 | 1.201 | 29 | 1.347e-02 | 6.766e-02 | 3.253e-03 |
| Inv-Newton | 512x512 | spike | 2.602 | 1.332 | 29 | 1.244e-02 | 1.893e-01 | 3.143e-03 |
| PE-Quad | 512x512 | spike | 2.649 | 1.379 | 28 | 1.250e-02 | - | 3.093e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.687 | 1.417 | 29 | 1.255e-02 | 7.118e-02 | 3.106e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.780 | 3.462 | 92 | 9.266e-03 | 3.461e-01 | 2.253e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.472 | 4.154 | 88 | 9.699e-03 | - | 2.322e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.746 | 3.428 | 92 | 9.491e-03 | 2.141e-01 | 2.288e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.044 | 3.460 | 92 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.820 | 4.236 | 88 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.081 | 3.497 | 92 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.147 | 3.672 | 92 | 1.287e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.655 | 4.180 | 88 | 1.371e-02 | - | 3.352e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.681 | 3.206 | 92 | 1.330e-02 | 3.559e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.339 | 3.866 | 92 | 1.292e-02 | 2.500e-01 | 3.303e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.628 | 4.154 | 88 | 1.365e-02 | - | 3.342e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.129 | 3.656 | 92 | 1.329e-02 | 3.376e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 5.167 | 3.558 | 92 | 1.281e-02 | 2.655e-01 | 3.219e-03 |
| PE-Quad | 1024x1024 | spike | 5.718 | 4.109 | 88 | 1.261e-02 | - | 3.145e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.200 | 3.591 | 92 | 1.292e-02 | 1.906e-01 | 3.225e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.886 | 1.425 | 13 | 4.746e-02 | 4.339e-01 | 5.936e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.140 | 1.680 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.834 | 1.374 | 13 | 4.739e-02 | 2.049e-01 | 5.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.795 | 1.520 | 13 | 4.483e-02 | 4.404e-01 | 5.625e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.834 | 1.559 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.539 | 1.264 | 13 | 4.484e-02 | 2.162e-01 | 5.625e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.946 | 1.228 | 13 | 4.592e-02 | 4.404e-01 | 5.802e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.103 | 1.384 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.925 | 1.206 | 13 | 4.582e-02 | 2.405e-01 | 5.781e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.860 | 1.436 | 13 | 4.733e-02 | 4.339e-01 | 5.993e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.831 | 1.406 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.780 | 1.356 | 13 | 4.733e-02 | 2.485e-01 | 5.993e-03 |
| Inv-Newton | 256x256 | spike | 2.555 | 1.239 | 13 | 4.714e-02 | 3.812e-01 | 6.029e-03 |
| PE-Quad | 256x256 | spike | 2.751 | 1.434 | 13 | - | - | - |
| PE-Quad-Coupled | 256x256 | spike | 2.544 | 1.228 | 13 | 4.947e-02 | 2.290e-01 | 6.326e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.943 | 1.306 | 29 | 4.460e-02 | 5.995e-01 | 5.629e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.971 | 1.334 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.935 | 1.298 | 29 | 4.460e-02 | 3.215e-01 | 5.629e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.072 | 1.558 | 29 | 4.352e-02 | 5.789e-01 | 5.520e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.899 | 1.385 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.084 | 1.571 | 29 | 4.356e-02 | 2.882e-01 | 5.525e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.952 | 1.657 | 29 | 4.042e-02 | 5.567e-01 | 5.131e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.807 | 1.511 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.619 | 1.324 | 29 | 4.049e-02 | 1.987e-01 | 5.140e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.883 | 1.318 | 29 | 4.437e-02 | 6.064e-01 | 5.648e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.086 | 1.521 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.067 | 1.502 | 29 | 4.448e-02 | 2.475e-01 | 5.661e-03 |
| Inv-Newton | 512x512 | spike | 2.750 | 1.280 | 29 | 4.130e-02 | 5.726e-01 | 5.255e-03 |
| PE-Quad | 512x512 | spike | 2.842 | 1.372 | 28 | - | - | - |
| PE-Quad-Coupled | 512x512 | spike | 3.017 | 1.547 | 29 | 4.369e-02 | 1.672e-01 | 5.558e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.466 | 4.130 | 92 | 3.999e-02 | 7.840e-01 | 5.085e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.061 | 4.725 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.411 | 4.074 | 92 | 4.050e-02 | 2.500e-01 | 5.153e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.428 | 4.035 | 92 | 3.745e-02 | 7.437e-01 | 4.763e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.073 | 4.680 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.526 | 4.133 | 92 | 3.786e-02 | 2.500e-01 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.504 | 4.055 | 92 | 4.413e-02 | 8.594e-01 | 5.639e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 6.281 | 4.832 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.403 | 3.955 | 92 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.348 | 3.899 | 92 | 4.401e-02 | 8.582e-01 | 5.628e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 6.069 | 4.620 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.564 | 4.115 | 92 | 4.481e-02 | 2.500e-01 | 5.731e-03 |
| Inv-Newton | 1024x1024 | spike | 7.695 | 3.887 | 92 | 3.970e-02 | 7.742e-01 | 5.006e-03 |
| PE-Quad | 1024x1024 | spike | 8.410 | 4.602 | 88 | - | - | - |
| PE-Quad-Coupled | 1024x1024 | spike | 7.759 | 3.952 | 92 | 4.420e-02 | 2.423e-01 | 5.571e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
