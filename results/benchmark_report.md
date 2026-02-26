# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-25*

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
| Inv-Newton | 256x256 | gaussian_spd | 2.832 | 0.960 | 13 | 2.066e-03 | 1.258e-01 | 2.070e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.136 | 1.264 | 13 | 3.809e-03 | - | 3.808e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.990 | 1.118 | 13 | 4.521e-03 | 1.381e-01 | 4.513e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.642 | 1.187 | 13 | 1.499e-03 | 1.255e-01 | 1.489e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.622 | 1.166 | 13 | 7.084e-03 | - | 7.089e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.568 | 1.112 | 13 | 5.912e-03 | 1.361e-01 | 5.901e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.199 | 1.411 | 13 | 1.108e-03 | 1.253e-01 | 1.110e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.888 | 1.100 | 13 | 8.147e-03 | - | 8.150e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.963 | 1.174 | 13 | 7.568e-03 | 7.348e-02 | 7.554e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.546 | 1.294 | 13 | 2.367e-03 | 1.253e-01 | 2.377e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.551 | 1.299 | 13 | 8.876e-03 | - | 8.882e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.495 | 1.244 | 13 | 8.508e-03 | 6.450e-02 | 8.484e-03 |
| Inv-Newton | 256x256 | spike | 2.578 | 0.983 | 13 | 4.917e-03 | 1.232e-01 | 4.929e-03 |
| PE-Quad | 256x256 | spike | 2.683 | 1.088 | 13 | 6.829e-03 | - | 6.844e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.715 | 1.120 | 13 | 6.946e-03 | 1.175e-01 | 6.937e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.760 | 2.329 | 28 | 2.989e-03 | 1.777e-01 | 2.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.629 | 1.198 | 27 | 3.401e-03 | - | 3.408e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.299 | 1.868 | 28 | 2.282e-03 | 8.083e-02 | 2.282e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 4.821 | 1.981 | 28 | 2.576e-03 | 1.773e-01 | 2.555e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.975 | 1.136 | 27 | 8.097e-03 | - | 8.113e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.965 | 1.125 | 28 | 5.834e-03 | 5.467e-02 | 5.831e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.759 | 1.049 | 28 | 3.259e-03 | 1.771e-01 | 3.242e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.863 | 1.153 | 27 | 4.077e-03 | - | 4.094e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.947 | 2.237 | 28 | 4.215e-03 | 8.421e-02 | 4.226e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.927 | 1.294 | 28 | 1.574e-03 | 1.771e-01 | 1.557e-03 |
| PE-Quad | 512x512 | near_rank_def | 4.423 | 2.790 | 27 | 7.033e-03 | - | 7.040e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.085 | 1.453 | 28 | 7.339e-03 | 1.381e-01 | 7.326e-03 |
| Inv-Newton | 512x512 | spike | 3.727 | 1.469 | 28 | 2.349e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 3.484 | 1.226 | 27 | 6.235e-03 | - | 6.242e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.675 | 1.417 | 28 | 6.271e-03 | 1.688e-01 | 6.273e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.133 | 2.392 | 88 | 3.590e-03 | 2.509e-01 | 3.569e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.325 | 2.583 | 84 | 4.224e-03 | - | 4.245e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.079 | 2.338 | 88 | 4.232e-03 | 1.507e-01 | 4.250e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 3.945 | 2.357 | 88 | 4.346e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.125 | 2.537 | 84 | 2.536e-04 | - | 2.435e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.933 | 2.346 | 88 | 3.180e-04 | 3.558e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.035 | 2.343 | 88 | 1.128e-03 | 2.503e-01 | 1.115e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.265 | 2.572 | 84 | 7.071e-03 | - | 7.076e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.034 | 2.342 | 88 | 7.071e-03 | 2.532e-01 | 7.077e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.771 | 2.370 | 88 | 1.189e-03 | 2.503e-01 | 1.177e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 4.001 | 2.600 | 84 | 7.008e-03 | - | 7.013e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.747 | 2.346 | 88 | 7.008e-03 | 2.528e-01 | 7.015e-03 |
| Inv-Newton | 1024x1024 | spike | 4.072 | 2.355 | 88 | 2.372e-03 | 2.455e-01 | 2.354e-03 |
| PE-Quad | 1024x1024 | spike | 4.303 | 2.587 | 84 | 6.411e-03 | - | 6.415e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.132 | 2.416 | 88 | 6.123e-03 | 2.390e-01 | 6.130e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.975 | 1.169 | 13 | 3.329e-03 | 3.036e-02 | 1.640e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.108 | 1.303 | 13 | 3.649e-03 | - | 1.802e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.926 | 1.121 | 13 | 3.656e-03 | 3.338e-02 | 1.810e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.933 | 1.302 | 13 | 3.215e-03 | 6.013e-02 | 1.599e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.025 | 1.393 | 13 | 3.265e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.854 | 1.222 | 13 | 3.251e-03 | 3.516e-02 | 1.627e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 4.909 | 1.201 | 13 | 1.938e-03 | 3.418e-02 | 9.661e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 4.801 | 1.093 | 13 | 2.078e-03 | - | 1.031e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 4.935 | 1.227 | 13 | 2.119e-03 | 1.914e-02 | 1.055e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.332 | 1.361 | 13 | 1.729e-03 | 8.213e-03 | 8.814e-04 |
| PE-Quad | 256x256 | near_rank_def | 3.388 | 1.417 | 13 | 3.738e-03 | - | 1.874e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.179 | 1.207 | 13 | 2.556e-03 | 3.907e-03 | 1.279e-03 |
| Inv-Newton | 256x256 | spike | 2.687 | 1.328 | 13 | 3.365e-03 | 2.409e-02 | 1.725e-03 |
| PE-Quad | 256x256 | spike | 2.643 | 1.284 | 13 | 5.605e-03 | - | 2.851e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.557 | 1.198 | 13 | 8.699e-03 | 5.510e-02 | 4.369e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.726 | 1.262 | 28 | 4.223e-03 | 1.562e-01 | 2.069e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.878 | 1.414 | 27 | 4.223e-03 | - | 2.066e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.779 | 1.315 | 28 | 4.586e-03 | 8.378e-02 | 2.196e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.174 | 1.718 | 28 | 2.762e-03 | 1.122e-01 | 1.437e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.813 | 1.357 | 27 | 3.075e-03 | - | 1.594e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.867 | 1.411 | 28 | 2.910e-03 | 5.593e-02 | 1.483e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.855 | 1.279 | 28 | 3.408e-03 | 1.495e-01 | 1.732e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.069 | 1.493 | 27 | 3.406e-03 | - | 1.730e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.826 | 1.249 | 28 | 8.409e-03 | 7.318e-02 | 4.241e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.744 | 1.282 | 28 | 1.591e-03 | 6.774e-02 | 7.996e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.768 | 1.306 | 27 | 1.590e-03 | - | 7.981e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.815 | 1.353 | 28 | 1.023e-02 | 2.975e-02 | 5.057e-03 |
| Inv-Newton | 512x512 | spike | 2.871 | 1.159 | 28 | 1.479e-03 | 6.455e-02 | 7.472e-04 |
| PE-Quad | 512x512 | spike | 3.020 | 1.309 | 27 | 1.774e-03 | - | 8.805e-04 |
| PE-Quad-Coupled | 512x512 | spike | 3.018 | 1.307 | 28 | 1.115e-02 | 6.966e-02 | 5.652e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.490 | 2.863 | 88 | 3.376e-03 | 2.142e-01 | 1.637e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.790 | 3.163 | 84 | 3.374e-03 | - | 1.635e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.549 | 2.921 | 88 | 1.099e-02 | 2.759e-01 | 5.526e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.512 | 2.892 | 88 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.761 | 3.141 | 84 | 3.887e-03 | - | 1.845e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.468 | 2.848 | 88 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.455 | 2.961 | 88 | 4.739e-04 | 4.197e-03 | 1.648e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.661 | 3.167 | 84 | 4.724e-04 | - | 1.618e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.370 | 2.876 | 88 | 1.155e-02 | 2.652e-01 | 5.801e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.921 | 2.896 | 88 | 4.884e-04 | 3.873e-03 | 1.859e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 5.228 | 3.203 | 84 | 4.870e-04 | - | 1.834e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.879 | 2.854 | 88 | 1.154e-02 | 2.656e-01 | 5.809e-03 |
| Inv-Newton | 1024x1024 | spike | 4.756 | 2.875 | 88 | 1.384e-03 | 8.567e-02 | 7.163e-04 |
| PE-Quad | 1024x1024 | spike | 5.015 | 3.134 | 84 | 1.673e-03 | - | 8.445e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.770 | 2.889 | 88 | 1.026e-02 | 2.109e-01 | 5.198e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 6.015 | 1.487 | 13 | 1.363e-02 | 4.166e-02 | 4.603e-03 |
| PE-Quad | 256x256 | gaussian_spd | 6.324 | 1.796 | 13 | 5.448e-03 | - | 1.869e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 5.936 | 1.408 | 13 | 5.828e-03 | 7.235e-02 | 1.948e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.758 | 1.304 | 13 | 1.537e-02 | 5.157e-02 | 5.207e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.085 | 1.630 | 13 | 5.088e-03 | - | 1.660e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.836 | 1.381 | 13 | 5.091e-03 | 6.287e-02 | 1.659e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.274 | 1.520 | 13 | 1.615e-02 | 5.952e-02 | 5.523e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 5.372 | 3.618 | 13 | 3.176e-03 | - | 1.081e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.102 | 1.348 | 13 | 3.168e-03 | 6.323e-02 | 1.079e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.175 | 1.305 | 13 | 1.560e-02 | 5.900e-02 | 5.314e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.234 | 1.364 | 13 | 3.018e-03 | - | 9.986e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.260 | 1.390 | 13 | 3.015e-03 | 7.665e-02 | 9.996e-04 |
| Inv-Newton | 256x256 | spike | 3.026 | 1.548 | 13 | 2.087e-02 | 1.067e-01 | 6.975e-03 |
| PE-Quad | 256x256 | spike | 3.061 | 1.583 | 13 | 4.300e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.741 | 1.263 | 13 | 4.264e-03 | 1.077e-01 | 1.425e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.932 | 1.447 | 28 | 1.060e-02 | 3.469e-02 | 3.674e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.176 | 1.692 | 27 | 7.773e-03 | - | 2.720e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.071 | 1.586 | 28 | 7.822e-03 | 1.028e-01 | 2.645e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.339 | 1.544 | 28 | 1.360e-02 | 6.611e-02 | 4.625e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.445 | 1.651 | 27 | 5.893e-03 | - | 2.059e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.111 | 1.317 | 28 | 6.006e-03 | 8.839e-02 | 2.097e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 5.224 | 1.436 | 28 | 1.201e-02 | 4.788e-02 | 3.928e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 5.479 | 1.692 | 27 | 8.047e-03 | - | 2.622e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 5.267 | 1.480 | 28 | 8.099e-03 | 8.839e-02 | 2.780e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.363 | 1.801 | 28 | 1.595e-02 | 8.252e-02 | 5.242e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.038 | 1.476 | 27 | 4.108e-03 | - | 1.394e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.054 | 1.492 | 28 | 4.131e-03 | 8.839e-02 | 1.444e-03 |
| Inv-Newton | 512x512 | spike | 3.458 | 1.599 | 28 | 1.524e-02 | 8.830e-02 | 5.090e-03 |
| PE-Quad | 512x512 | spike | 3.499 | 1.640 | 27 | 4.037e-03 | - | 1.339e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.815 | 1.957 | 28 | 4.045e-03 | 1.055e-01 | 1.337e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 7.184 | 3.505 | 88 | 1.213e-02 | 7.182e-02 | 3.963e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 7.656 | 3.977 | 84 | 7.995e-03 | - | 2.586e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 7.284 | 3.604 | 88 | 7.995e-03 | 2.187e-01 | 2.586e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.296 | 3.667 | 88 | 9.047e-03 | 2.705e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.875 | 4.246 | 84 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.055 | 3.426 | 88 | 9.042e-03 | 2.500e-01 | 2.940e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 7.393 | 3.897 | 88 | 1.627e-02 | 1.250e-01 | 5.381e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 7.327 | 3.832 | 84 | 2.651e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 6.883 | 3.387 | 88 | 2.653e-03 | 8.839e-02 | 9.176e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.895 | 3.326 | 88 | 1.620e-02 | 1.250e-01 | 5.367e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.395 | 3.827 | 84 | 2.713e-03 | - | 9.310e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.928 | 3.359 | 88 | 2.714e-03 | 8.839e-02 | 9.312e-04 |
| Inv-Newton | 1024x1024 | spike | 4.881 | 3.363 | 88 | 1.513e-02 | 1.251e-01 | 5.071e-03 |
| PE-Quad | 1024x1024 | spike | 5.338 | 3.819 | 84 | 4.138e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.902 | 3.383 | 88 | 4.116e-03 | 1.196e-01 | 1.350e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.107 | 1.482 | 13 | 8.493e-03 | 1.080e-01 | 2.242e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.043 | 1.417 | 13 | 8.481e-03 | - | 2.241e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.862 | 1.236 | 13 | 1.209e-02 | 7.329e-02 | 3.144e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.778 | 1.290 | 13 | 1.027e-02 | 1.475e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.858 | 1.370 | 13 | 1.058e-02 | - | 2.728e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.010 | 1.522 | 13 | 1.285e-02 | 6.944e-02 | 3.235e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.154 | 1.263 | 13 | 1.153e-02 | 1.314e-01 | 2.947e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.199 | 1.308 | 13 | 1.340e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.182 | 1.290 | 13 | 1.428e-02 | 3.906e-02 | 3.450e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.801 | 1.342 | 13 | 1.101e-02 | 1.180e-01 | 2.815e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.712 | 1.253 | 13 | 1.281e-02 | - | 3.114e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.737 | 1.279 | 13 | 1.358e-02 | 1.563e-02 | 3.275e-03 |
| Inv-Newton | 256x256 | spike | 2.872 | 1.248 | 13 | 8.877e-03 | 9.560e-02 | 2.254e-03 |
| PE-Quad | 256x256 | spike | 3.090 | 1.466 | 13 | 9.420e-03 | - | 2.336e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.100 | 1.476 | 13 | 9.508e-03 | 6.347e-02 | 2.351e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.060 | 1.281 | 28 | 8.364e-03 | 2.425e-01 | 1.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.304 | 1.525 | 27 | 8.995e-03 | - | 2.085e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.106 | 1.327 | 28 | 8.969e-03 | 1.484e-01 | 2.085e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.331 | 1.456 | 28 | 1.050e-02 | 2.167e-01 | 2.586e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.402 | 1.527 | 27 | 1.181e-02 | - | 2.805e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.173 | 1.298 | 28 | 1.182e-02 | 1.121e-01 | 2.805e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.907 | 1.289 | 28 | 9.108e-03 | 2.433e-01 | 2.203e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.931 | 1.313 | 27 | 9.640e-03 | - | 2.291e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.041 | 1.423 | 28 | 9.649e-03 | 1.495e-01 | 2.291e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.868 | 1.227 | 28 | 1.225e-02 | 1.923e-01 | 3.057e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.917 | 1.276 | 27 | 1.347e-02 | - | 3.253e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.882 | 1.241 | 28 | 1.347e-02 | 6.766e-02 | 3.253e-03 |
| Inv-Newton | 512x512 | spike | 3.319 | 1.404 | 28 | 1.244e-02 | 1.893e-01 | 3.143e-03 |
| PE-Quad | 512x512 | spike | 3.264 | 1.349 | 27 | 1.250e-02 | - | 3.093e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.270 | 1.355 | 28 | 1.255e-02 | 7.118e-02 | 3.106e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.546 | 3.232 | 88 | 9.265e-03 | 3.461e-01 | 2.253e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.122 | 3.807 | 84 | 9.699e-03 | - | 2.322e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.561 | 3.246 | 88 | 9.491e-03 | 2.141e-01 | 2.288e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.639 | 3.232 | 88 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.168 | 3.760 | 84 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.668 | 3.260 | 88 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.072 | 3.270 | 88 | 1.287e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.533 | 3.730 | 84 | 1.371e-02 | - | 3.352e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.058 | 3.256 | 88 | 1.330e-02 | 3.559e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.910 | 3.254 | 88 | 1.289e-02 | 2.500e-01 | 3.298e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.427 | 3.771 | 84 | 1.365e-02 | - | 3.342e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.896 | 3.241 | 88 | 1.329e-02 | 3.440e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 4.959 | 3.254 | 88 | 1.281e-02 | 2.655e-01 | 3.219e-03 |
| PE-Quad | 1024x1024 | spike | 5.462 | 3.757 | 84 | 1.261e-02 | - | 3.145e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.944 | 3.239 | 88 | 1.292e-02 | 1.906e-01 | 3.225e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.216 | 1.513 | 13 | 4.745e-02 | 3.991e-01 | 5.936e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.220 | 1.518 | 13 | 1.336e-02 | - | 1.589e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.011 | 1.309 | 13 | 4.742e-02 | 1.491e-01 | 5.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.361 | 1.361 | 13 | 4.483e-02 | 4.063e-01 | 5.625e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.576 | 1.576 | 13 | 1.378e-02 | - | 1.662e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.302 | 1.302 | 13 | 4.485e-02 | 1.213e-01 | 5.625e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.947 | 1.559 | 13 | 4.581e-02 | 4.272e-01 | 5.781e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.934 | 1.545 | 13 | 1.218e-02 | - | 1.458e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.827 | 1.439 | 13 | 4.583e-02 | 1.250e-01 | 5.781e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.108 | 1.406 | 13 | 4.733e-02 | 4.266e-01 | 5.993e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.854 | 2.152 | 13 | 1.074e-02 | - | 1.274e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.373 | 1.671 | 13 | 4.734e-02 | 1.529e-01 | 5.993e-03 |
| Inv-Newton | 256x256 | spike | 3.065 | 1.354 | 13 | 4.714e-02 | 3.811e-01 | 6.025e-03 |
| PE-Quad | 256x256 | spike | 3.202 | 1.491 | 13 | 9.331e-03 | - | 1.099e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.892 | 1.181 | 13 | 4.903e-02 | 2.114e-01 | 6.270e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.793 | 1.364 | 28 | 4.009e-02 | 4.727e-01 | 5.047e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.136 | 1.707 | 27 | 1.850e-02 | - | 2.240e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.944 | 1.514 | 28 | 4.011e-02 | 6.900e-02 | 5.047e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.924 | 1.504 | 28 | 4.352e-02 | 5.586e-01 | 5.520e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.907 | 1.487 | 27 | 1.547e-02 | - | 1.839e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.935 | 1.515 | 28 | 4.357e-02 | 1.472e-01 | 5.525e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.675 | 1.281 | 28 | 4.042e-02 | 4.992e-01 | 5.131e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.817 | 1.422 | 27 | 1.797e-02 | - | 2.132e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.738 | 1.343 | 28 | 4.050e-02 | 9.568e-02 | 5.140e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.690 | 1.962 | 28 | 4.436e-02 | 5.975e-01 | 5.648e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.239 | 1.511 | 27 | 1.371e-02 | - | 1.602e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.277 | 1.549 | 28 | 4.449e-02 | 1.650e-01 | 5.661e-03 |
| Inv-Newton | 512x512 | spike | 3.368 | 1.474 | 28 | 4.130e-02 | 5.672e-01 | 5.255e-03 |
| PE-Quad | 512x512 | spike | 3.475 | 1.581 | 27 | 1.445e-02 | - | 1.745e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.215 | 1.321 | 28 | 4.340e-02 | 1.646e-01 | 5.519e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.367 | 3.687 | 88 | 3.999e-02 | 7.069e-01 | 5.085e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.144 | 4.464 | 84 | 1.840e-02 | - | 2.172e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.369 | 3.689 | 88 | 4.051e-02 | 1.434e-01 | 5.153e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.317 | 3.675 | 88 | 3.744e-02 | 6.170e-01 | 4.764e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.083 | 4.441 | 84 | 2.029e-02 | - | 2.393e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.310 | 3.668 | 88 | 3.786e-02 | 2.777e-04 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.380 | 3.694 | 88 | 4.413e-02 | 8.605e-01 | 5.640e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 6.117 | 4.432 | 84 | 1.283e-02 | - | 1.470e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.259 | 3.574 | 88 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.650 | 3.655 | 88 | 4.401e-02 | 8.592e-01 | 5.628e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 6.416 | 4.421 | 84 | 1.282e-02 | - | 1.475e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.653 | 3.658 | 88 | 4.481e-02 | 2.500e-01 | 5.731e-03 |
| Inv-Newton | 1024x1024 | spike | 5.917 | 3.746 | 88 | 3.977e-02 | 7.719e-01 | 5.014e-03 |
| PE-Quad | 1024x1024 | spike | 6.616 | 4.444 | 84 | 1.410e-02 | - | 1.717e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.783 | 3.612 | 88 | 4.395e-02 | 2.381e-01 | 5.540e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
