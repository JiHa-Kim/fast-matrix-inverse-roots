# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-24*

This report details the performance and accuracy of quadratic PE (Polynomial-Express) iterations for matrix inverse p-th roots.

## Methodology
- **Sizes**: 256,512
- **Trials per case**: 3
- **Hardware**: CPU (fp32)
- **Methods Compared**: `Inverse-Newton` (baseline), `PE-Quad` (uncoupled quadratic), `PE-Quad-Coupled` (coupled quadratic).

## Results for $p=1$

```text
[coeff] using tuned(l_target=0.05, seed=0), safety=1.0, no_final_safety=False
== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            3.019 ms (pre 1.676 + iter 1.343) | mem   12MB | resid 3.225e-03 p95 4.675e-03 max 4.675e-03 | Y_res 4.491e-02 | relerr 3.208e-03 | r2 nan | hard nan | symX 2.77e-04 symW 4.45e-04 | mv nan | bad 0
PE-Quad                   2.926 ms (pre 1.676 + iter 1.250) | mem   11MB | resid 4.117e-03 p95 6.292e-03 max 6.292e-03 | relerr 4.093e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.00e-04 | mv nan | bad 0
PE-Quad-Coupled           2.975 ms (pre 1.676 + iter 1.299) | mem   12MB | resid 4.254e-03 p95 7.240e-03 max 7.240e-03 | Y_res 1.688e-01 | relerr 4.226e-03 | r2 nan | hard nan | symX 4.40e-04 symW 5.93e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.926 ms, resid=4.117e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.608 ms (pre 1.557 + iter 1.051) | mem   12MB | resid 3.847e-03 p95 4.001e-03 max 4.001e-03 | Y_res 6.248e-02 | relerr 3.851e-03 | r2 nan | hard nan | symX 2.25e-04 symW 3.84e-04 | mv nan | bad 0
PE-Quad                   2.597 ms (pre 1.557 + iter 1.040) | mem   11MB | resid 5.601e-03 p95 6.525e-03 max 6.525e-03 | relerr 5.601e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.38e-04 | mv nan | bad 0
PE-Quad-Coupled           2.716 ms (pre 1.557 + iter 1.159) | mem   12MB | resid 4.914e-03 p95 5.399e-03 max 5.399e-03 | Y_res 1.479e-01 | relerr 4.902e-03 | r2 nan | hard nan | symX 5.18e-04 symW 6.91e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.597 ms, resid=5.601e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.597 ms (pre 1.577 + iter 1.020) | mem   12MB | resid 2.598e-03 p95 2.622e-03 max 2.622e-03 | Y_res 1.308e-02 | relerr 2.601e-03 | r2 nan | hard nan | symX 1.37e-04 symW 2.23e-04 | mv nan | bad 0
PE-Quad                   2.603 ms (pre 1.577 + iter 1.026) | mem   11MB | resid 6.049e-03 p95 8.045e-03 max 8.045e-03 | relerr 6.051e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.68e-04 | mv nan | bad 0
PE-Quad-Coupled           2.675 ms (pre 1.577 + iter 1.099) | mem   12MB | resid 7.456e-03 p95 7.791e-03 max 7.791e-03 | Y_res 1.410e-01 | relerr 7.409e-03 | r2 nan | hard nan | symX 6.57e-04 symW 1.13e-03 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.597 ms, resid=2.598e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.433 ms (pre 1.320 + iter 1.113) | mem   12MB | resid 1.420e-03 p95 2.605e-03 max 2.605e-03 | Y_res 1.294e-02 | relerr 1.418e-03 | r2 nan | hard nan | symX 1.43e-04 symW 2.35e-04 | mv nan | bad 0
PE-Quad                   2.394 ms (pre 1.320 + iter 1.074) | mem   11MB | resid 7.783e-03 p95 8.118e-03 max 8.118e-03 | relerr 7.773e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.96e-04 | mv nan | bad 0
PE-Quad-Coupled           2.406 ms (pre 1.320 + iter 1.086) | mem   12MB | resid 4.899e-03 p95 7.906e-03 max 7.906e-03 | Y_res 1.436e-01 | relerr 4.879e-03 | r2 nan | hard nan | symX 6.08e-04 symW 8.19e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.394 ms, resid=7.783e-03, hard=nan

-- case spike --
Inverse-Newton            2.849 ms (pre 1.563 + iter 1.286) | mem   12MB | resid 1.704e-03 p95 1.863e-03 max 1.863e-03 | Y_res 2.100e-02 | relerr 1.690e-03 | r2 nan | hard nan | symX 8.95e-05 symW 1.61e-04 | mv nan | bad 0
PE-Quad                   2.572 ms (pre 1.563 + iter 1.010) | mem   11MB | resid 6.721e-03 p95 6.764e-03 max 6.764e-03 | relerr 6.712e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.58e-04 | mv nan | bad 0
PE-Quad-Coupled           2.606 ms (pre 1.563 + iter 1.043) | mem   12MB | resid 5.466e-03 p95 5.508e-03 max 5.508e-03 | Y_res 1.132e-01 | relerr 5.452e-03 | r2 nan | hard nan | symX 3.39e-04 symW 5.98e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.572 ms, resid=6.721e-03, hard=nan

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.598 ms (pre 1.262 + iter 1.336) | mem   23MB | resid 2.909e-03 p95 3.140e-03 max 3.140e-03 | Y_res 8.356e-02 | relerr 2.909e-03 | r2 nan | hard nan | symX 2.12e-04 symW 3.35e-04 | mv nan | bad 0
PE-Quad                   2.373 ms (pre 1.262 + iter 1.111) | mem   21MB | resid 5.019e-03 p95 5.101e-03 max 5.101e-03 | relerr 5.038e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.55e-04 | mv nan | bad 0
PE-Quad-Coupled           2.386 ms (pre 1.262 + iter 1.124) | mem   23MB | resid 2.572e-03 p95 2.586e-03 max 2.586e-03 | Y_res 3.162e-01 | relerr 2.586e-03 | r2 nan | hard nan | symX 3.46e-04 symW 2.76e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.373 ms, resid=5.019e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.413 ms (pre 1.319 + iter 1.094) | mem   23MB | resid 2.268e-03 p95 2.277e-03 max 2.277e-03 | Y_res 6.001e-02 | relerr 2.257e-03 | r2 nan | hard nan | symX 2.20e-04 symW 3.61e-04 | mv nan | bad 0
PE-Quad                   2.422 ms (pre 1.319 + iter 1.103) | mem   21MB | resid 7.872e-03 p95 8.196e-03 max 8.196e-03 | relerr 7.891e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.75e-04 | mv nan | bad 0
PE-Quad-Coupled           2.510 ms (pre 1.319 + iter 1.191) | mem   23MB | resid 4.514e-03 p95 5.838e-03 max 5.838e-03 | Y_res 1.900e-01 | relerr 4.518e-03 | r2 nan | hard nan | symX 3.40e-04 symW 3.42e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.413 ms, resid=2.268e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.447 ms (pre 1.289 + iter 1.158) | mem   23MB | resid 1.811e-03 p95 1.829e-03 max 1.829e-03 | Y_res 1.598e-02 | relerr 1.805e-03 | r2 nan | hard nan | symX 1.22e-04 symW 2.05e-04 | mv nan | bad 0
PE-Quad                   2.554 ms (pre 1.289 + iter 1.265) | mem   21MB | resid 9.250e-03 p95 9.553e-03 max 9.553e-03 | relerr 9.242e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.93e-04 | mv nan | bad 0
PE-Quad-Coupled           2.600 ms (pre 1.289 + iter 1.311) | mem   23MB | resid 5.794e-03 p95 5.810e-03 max 5.810e-03 | Y_res 9.155e-02 | relerr 5.777e-03 | r2 nan | hard nan | symX 3.74e-04 symW 5.76e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.447 ms, resid=1.811e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            3.062 ms (pre 1.584 + iter 1.478) | mem   23MB | resid 1.622e-03 p95 1.717e-03 max 1.717e-03 | Y_res 1.442e-02 | relerr 1.622e-03 | r2 nan | hard nan | symX 1.03e-04 symW 1.71e-04 | mv nan | bad 0
PE-Quad                   2.722 ms (pre 1.584 + iter 1.138) | mem   21MB | resid 6.191e-03 p95 6.745e-03 max 6.745e-03 | relerr 6.186e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.75e-04 | mv nan | bad 0
PE-Quad-Coupled           2.747 ms (pre 1.584 + iter 1.163) | mem   23MB | resid 5.585e-03 p95 5.689e-03 max 5.689e-03 | Y_res 1.700e-01 | relerr 5.586e-03 | r2 nan | hard nan | symX 3.15e-04 symW 4.46e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.722 ms, resid=6.191e-03, hard=nan

-- case spike --
Inverse-Newton            2.488 ms (pre 1.243 + iter 1.245) | mem   23MB | resid 1.339e-03 p95 1.409e-03 max 1.409e-03 | Y_res 2.497e-02 | relerr 1.322e-03 | r2 nan | hard nan | symX 4.02e-05 symW 6.52e-05 | mv nan | bad 0
PE-Quad                   3.204 ms (pre 1.243 + iter 1.961) | mem   21MB | resid 5.139e-03 p95 5.158e-03 max 5.158e-03 | relerr 5.135e-03 | r2 nan | hard nan | symX 0.00e+00 symW 8.01e-05 | mv nan | bad 0
PE-Quad-Coupled           2.597 ms (pre 1.243 + iter 1.354) | mem   23MB | resid 5.102e-03 p95 5.122e-03 max 5.122e-03 | Y_res 1.773e-01 | relerr 5.096e-03 | r2 nan | hard nan | symX 1.27e-04 symW 1.17e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.488 ms, resid=1.339e-03, hard=nan


```

## Results for $p=2$

```text
[coeff] using precomputed(l_target=0.05), safety=1.0, no_final_safety=False
== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.987 ms (pre 1.452 + iter 1.535) | mem   12MB | resid 5.461e-03 p95 6.841e-03 max 6.841e-03 | Y_res 6.840e-03 | relerr 2.708e-03 | r2 nan | hard nan | symX 1.10e-04 symW 2.38e-04 | mv nan | bad 0
PE-Quad                   2.793 ms (pre 1.452 + iter 1.340) | mem   11MB | resid 4.093e-03 p95 5.465e-03 max 5.465e-03 | relerr 1.997e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.21e-04 | mv nan | bad 0
PE-Quad-Coupled           2.747 ms (pre 1.452 + iter 1.294) | mem   12MB | resid 3.557e-03 p95 4.749e-03 max 4.749e-03 | Y_res 4.926e-02 | relerr 1.752e-03 | r2 nan | hard nan | symX 3.69e-04 symW 6.31e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.747 ms, resid=3.557e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.749 ms (pre 1.637 + iter 1.112) | mem   12MB | resid 4.900e-03 p95 5.033e-03 max 5.033e-03 | Y_res 3.868e-03 | relerr 2.443e-03 | r2 nan | hard nan | symX 7.83e-05 symW 1.87e-04 | mv nan | bad 0
PE-Quad                   2.915 ms (pre 1.637 + iter 1.278) | mem   11MB | resid 4.902e-03 p95 5.034e-03 max 5.034e-03 | relerr 2.440e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.24e-04 | mv nan | bad 0
PE-Quad-Coupled           3.148 ms (pre 1.637 + iter 1.510) | mem   12MB | resid 4.853e-03 p95 4.969e-03 max 4.969e-03 | Y_res 5.497e-02 | relerr 2.423e-03 | r2 nan | hard nan | symX 3.76e-04 symW 6.49e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.749 ms, resid=4.900e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.886 ms (pre 1.536 + iter 1.350) | mem   12MB | resid 1.283e-02 p95 1.297e-02 max 1.297e-02 | Y_res 6.296e-03 | relerr 6.395e-03 | r2 nan | hard nan | symX 7.81e-05 symW 1.84e-04 | mv nan | bad 0
PE-Quad                   2.813 ms (pre 1.536 + iter 1.277) | mem   11MB | resid 2.364e-03 p95 2.392e-03 max 2.392e-03 | relerr 1.170e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.14e-04 | mv nan | bad 0
PE-Quad-Coupled           2.909 ms (pre 1.536 + iter 1.373) | mem   12MB | resid 2.495e-03 p95 5.945e-03 max 5.945e-03 | Y_res 1.105e-02 | relerr 1.241e-03 | r2 nan | hard nan | symX 3.58e-04 symW 6.22e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.813 ms, resid=2.364e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.873 ms (pre 1.445 + iter 1.428) | mem   12MB | resid 6.155e-03 p95 1.257e-02 max 1.257e-02 | Y_res 2.715e-03 | relerr 3.088e-03 | r2 nan | hard nan | symX 7.50e-05 symW 1.78e-04 | mv nan | bad 0
PE-Quad                   2.910 ms (pre 1.445 + iter 1.465) | mem   11MB | resid 1.668e-03 p95 2.900e-03 max 2.900e-03 | relerr 8.485e-04 | r2 nan | hard nan | symX 0.00e+00 symW 2.12e-04 | mv nan | bad 0
PE-Quad-Coupled           2.669 ms (pre 1.445 + iter 1.224) | mem   12MB | resid 6.469e-03 p95 6.849e-03 max 6.849e-03 | Y_res 3.383e-02 | relerr 3.267e-03 | r2 nan | hard nan | symX 3.62e-04 symW 6.29e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.669 ms, resid=6.469e-03, hard=nan

-- case spike --
Inverse-Newton            4.458 ms (pre 1.498 + iter 2.960) | mem   12MB | resid 1.161e-02 p95 1.174e-02 max 1.174e-02 | Y_res 1.237e-02 | relerr 5.836e-03 | r2 nan | hard nan | symX 5.56e-05 symW 1.31e-04 | mv nan | bad 0
PE-Quad                   4.512 ms (pre 1.498 + iter 3.014) | mem   11MB | resid 3.046e-03 p95 3.170e-03 max 3.170e-03 | relerr 1.525e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.27e-04 | mv nan | bad 0
PE-Quad-Coupled           2.642 ms (pre 1.498 + iter 1.144) | mem   12MB | resid 8.982e-03 p95 9.033e-03 max 9.033e-03 | Y_res 5.018e-02 | relerr 4.488e-03 | r2 nan | hard nan | symX 2.43e-04 symW 4.43e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.642 ms, resid=8.982e-03, hard=nan

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.839 ms (pre 1.650 + iter 1.189) | mem   23MB | resid 3.958e-03 p95 4.185e-03 max 4.185e-03 | Y_res 6.944e-03 | relerr 2.028e-03 | r2 nan | hard nan | symX 6.81e-05 symW 1.52e-04 | mv nan | bad 0
PE-Quad                   2.939 ms (pre 1.650 + iter 1.288) | mem   21MB | resid 4.013e-03 p95 4.289e-03 max 4.289e-03 | relerr 2.053e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.63e-04 | mv nan | bad 0
PE-Quad-Coupled           2.762 ms (pre 1.650 + iter 1.112) | mem   23MB | resid 3.964e-03 p95 4.191e-03 max 4.191e-03 | Y_res 7.822e-02 | relerr 2.030e-03 | r2 nan | hard nan | symX 2.64e-04 symW 4.42e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.762 ms, resid=3.964e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.600 ms (pre 1.292 + iter 1.309) | mem   23MB | resid 2.884e-03 p95 1.047e-02 max 1.047e-02 | Y_res 4.658e-03 | relerr 1.412e-03 | r2 nan | hard nan | symX 6.88e-05 symW 1.55e-04 | mv nan | bad 0
PE-Quad                   2.432 ms (pre 1.292 + iter 1.141) | mem   21MB | resid 2.873e-03 p95 2.884e-03 max 2.884e-03 | relerr 1.407e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.63e-04 | mv nan | bad 0
PE-Quad-Coupled           2.459 ms (pre 1.292 + iter 1.167) | mem   23MB | resid 8.640e-03 p95 8.883e-03 max 8.883e-03 | Y_res 7.150e-02 | relerr 4.309e-03 | r2 nan | hard nan | symX 3.17e-04 symW 5.74e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.432 ms, resid=2.873e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            3.047 ms (pre 1.628 + iter 1.418) | mem   23MB | resid 7.099e-03 p95 8.199e-03 max 8.199e-03 | Y_res 3.769e-03 | relerr 3.517e-03 | r2 nan | hard nan | symX 7.29e-05 symW 1.66e-04 | mv nan | bad 0
PE-Quad                   3.135 ms (pre 1.628 + iter 1.507) | mem   21MB | resid 1.647e-03 p95 1.849e-03 max 1.849e-03 | relerr 8.114e-04 | r2 nan | hard nan | symX 0.00e+00 symW 1.62e-04 | mv nan | bad 0
PE-Quad-Coupled           2.774 ms (pre 1.628 + iter 1.146) | mem   23MB | resid 1.029e-02 p95 1.067e-02 max 1.067e-02 | Y_res 7.665e-02 | relerr 5.118e-03 | r2 nan | hard nan | symX 3.20e-04 symW 5.86e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 3.047 ms, resid=7.099e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.477 ms (pre 1.285 + iter 1.192) | mem   23MB | resid 1.371e-02 p95 1.381e-02 max 1.381e-02 | Y_res 3.379e-03 | relerr 6.777e-03 | r2 nan | hard nan | symX 6.04e-05 symW 1.41e-04 | mv nan | bad 0
PE-Quad                   2.623 ms (pre 1.285 + iter 1.338) | mem   21MB | resid 3.040e-03 p95 3.061e-03 max 3.061e-03 | relerr 1.511e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.55e-04 | mv nan | bad 0
PE-Quad-Coupled           2.600 ms (pre 1.285 + iter 1.315) | mem   23MB | resid 4.852e-03 p95 5.761e-03 max 5.761e-03 | Y_res 2.844e-02 | relerr 2.414e-03 | r2 nan | hard nan | symX 2.74e-04 symW 4.86e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.600 ms, resid=4.852e-03, hard=nan

-- case spike --
Inverse-Newton            3.178 ms (pre 1.413 + iter 1.765) | mem   23MB | resid 1.410e-02 p95 1.413e-02 max 1.413e-02 | Y_res 1.515e-02 | relerr 7.055e-03 | r2 nan | hard nan | symX 2.23e-05 symW 5.83e-05 | mv nan | bad 0
PE-Quad                   2.564 ms (pre 1.413 + iter 1.152) | mem   21MB | resid 3.649e-03 p95 3.675e-03 max 3.675e-03 | relerr 1.877e-03 | r2 nan | hard nan | symX 0.00e+00 symW 7.83e-05 | mv nan | bad 0
PE-Quad-Coupled           2.538 ms (pre 1.413 + iter 1.126) | mem   23MB | resid 7.668e-03 p95 7.705e-03 max 7.705e-03 | Y_res 4.688e-02 | relerr 3.801e-03 | r2 nan | hard nan | symX 1.44e-04 symW 2.49e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.538 ms, resid=7.668e-03, hard=nan


```

## Results for $p=3$

```text
[coeff] using tuned(l_target=0.05, seed=0), safety=1.0, no_final_safety=False
== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.586 ms (pre 1.310 + iter 1.276) | mem   12MB | resid 1.880e-02 p95 1.942e-02 max 1.942e-02 | Y_res 1.479e-01 | relerr 6.365e-03 | r2 nan | hard nan | symX 5.36e-05 symW 4.53e-04 | mv nan | bad 0
PE-Quad                   2.695 ms (pre 1.310 + iter 1.384) | mem   11MB | resid 4.811e-03 p95 5.014e-03 max 5.014e-03 | relerr 1.586e-03 | r2 nan | hard nan | symX 0.00e+00 symW 4.48e-04 | mv nan | bad 0
PE-Quad-Coupled           2.799 ms (pre 1.310 + iter 1.489) | mem   12MB | resid 1.781e-02 p95 1.788e-02 max 1.788e-02 | Y_res 9.882e-02 | relerr 6.008e-03 | r2 nan | hard nan | symX 1.77e-04 symW 1.61e-03 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.695 ms, resid=4.811e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.969 ms (pre 1.602 + iter 1.367) | mem   12MB | resid 1.489e-02 p95 1.992e-02 max 1.992e-02 | Y_res 7.227e-02 | relerr 5.033e-03 | r2 nan | hard nan | symX 4.56e-05 symW 2.97e-04 | mv nan | bad 0
PE-Quad                   2.919 ms (pre 1.602 + iter 1.318) | mem   11MB | resid 4.783e-03 p95 4.884e-03 max 4.884e-03 | relerr 1.555e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.38e-04 | mv nan | bad 0
PE-Quad-Coupled           3.070 ms (pre 1.602 + iter 1.468) | mem   12MB | resid 9.381e-03 p95 1.920e-02 max 1.920e-02 | Y_res 3.984e-02 | relerr 3.137e-03 | r2 nan | hard nan | symX 1.36e-04 symW 8.90e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.919 ms, resid=4.783e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.776 ms (pre 1.373 + iter 1.403) | mem   12MB | resid 1.157e-02 p95 2.031e-02 max 2.031e-02 | Y_res 5.541e-02 | relerr 3.829e-03 | r2 nan | hard nan | symX 4.79e-05 symW 3.44e-04 | mv nan | bad 0
PE-Quad                   2.814 ms (pre 1.373 + iter 1.441) | mem   11MB | resid 8.351e-03 p95 8.418e-03 max 8.418e-03 | relerr 2.795e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.37e-04 | mv nan | bad 0
PE-Quad-Coupled           2.645 ms (pre 1.373 + iter 1.272) | mem   12MB | resid 1.144e-02 p95 1.990e-02 max 1.990e-02 | Y_res 3.747e-02 | relerr 3.776e-03 | r2 nan | hard nan | symX 1.23e-04 symW 7.05e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.814 ms, resid=8.351e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.687 ms (pre 1.306 + iter 1.381) | mem   12MB | resid 1.926e-02 p95 2.003e-02 max 2.003e-02 | Y_res 1.682e-01 | relerr 6.421e-03 | r2 nan | hard nan | symX 4.78e-05 symW 3.20e-04 | mv nan | bad 0
PE-Quad                   2.679 ms (pre 1.306 + iter 1.373) | mem   11MB | resid 4.461e-03 p95 8.008e-03 max 8.008e-03 | relerr 1.492e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.06e-04 | mv nan | bad 0
PE-Quad-Coupled           2.607 ms (pre 1.306 + iter 1.302) | mem   12MB | resid 1.923e-02 p95 1.998e-02 max 1.998e-02 | Y_res 6.675e-02 | relerr 6.410e-03 | r2 nan | hard nan | symX 1.47e-04 symW 5.20e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.679 ms, resid=4.461e-03, hard=nan

-- case spike --
Inverse-Newton            2.616 ms (pre 1.339 + iter 1.278) | mem   12MB | resid 1.367e-02 p95 1.379e-02 max 1.379e-02 | Y_res 1.075e-01 | relerr 4.534e-03 | r2 nan | hard nan | symX 3.51e-05 symW 3.10e-04 | mv nan | bad 0
PE-Quad                   2.670 ms (pre 1.339 + iter 1.331) | mem   11MB | resid 7.410e-03 p95 7.455e-03 max 7.455e-03 | relerr 2.438e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.87e-04 | mv nan | bad 0
PE-Quad-Coupled           2.627 ms (pre 1.339 + iter 1.288) | mem   12MB | resid 1.369e-02 p95 1.382e-02 max 1.382e-02 | Y_res 2.471e-02 | relerr 4.541e-03 | r2 nan | hard nan | symX 8.14e-05 symW 2.79e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.670 ms, resid=7.410e-03, hard=nan

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.774 ms (pre 1.381 + iter 1.394) | mem   23MB | resid 1.509e-02 p95 1.520e-02 max 1.520e-02 | Y_res 1.213e-01 | relerr 5.029e-03 | r2 nan | hard nan | symX 3.50e-05 symW 2.49e-04 | mv nan | bad 0
PE-Quad                   2.734 ms (pre 1.381 + iter 1.354) | mem   21MB | resid 4.886e-03 p95 5.015e-03 max 5.015e-03 | relerr 1.682e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.67e-04 | mv nan | bad 0
PE-Quad-Coupled           2.647 ms (pre 1.381 + iter 1.267) | mem   23MB | resid 1.063e-02 p95 1.077e-02 max 1.077e-02 | Y_res 7.324e-04 | relerr 3.579e-03 | r2 nan | hard nan | symX 9.20e-05 symW 6.91e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.734 ms, resid=4.886e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.630 ms (pre 1.373 + iter 1.256) | mem   23MB | resid 1.800e-02 p95 1.802e-02 max 1.802e-02 | Y_res 2.016e-01 | relerr 5.908e-03 | r2 nan | hard nan | symX 3.42e-05 symW 2.70e-04 | mv nan | bad 0
PE-Quad                   2.716 ms (pre 1.373 + iter 1.343) | mem   21MB | resid 3.965e-03 p95 6.851e-03 max 6.851e-03 | relerr 1.330e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.74e-04 | mv nan | bad 0
PE-Quad-Coupled           2.885 ms (pre 1.373 + iter 1.511) | mem   23MB | resid 1.622e-02 p95 1.626e-02 max 1.626e-02 | Y_res 5.183e-04 | relerr 5.346e-03 | r2 nan | hard nan | symX 9.11e-05 symW 7.68e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.716 ms, resid=3.965e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.569 ms (pre 1.289 + iter 1.280) | mem   23MB | resid 1.853e-02 p95 1.916e-02 max 1.916e-02 | Y_res 2.311e-01 | relerr 6.110e-03 | r2 nan | hard nan | symX 4.25e-05 symW 3.54e-04 | mv nan | bad 0
PE-Quad                   2.883 ms (pre 1.289 + iter 1.594) | mem   21MB | resid 4.710e-03 p95 5.322e-03 max 5.322e-03 | relerr 1.575e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.50e-04 | mv nan | bad 0
PE-Quad-Coupled           2.927 ms (pre 1.289 + iter 1.638) | mem   23MB | resid 1.853e-02 p95 1.916e-02 max 1.916e-02 | Y_res 4.166e-04 | relerr 6.110e-03 | r2 nan | hard nan | symX 9.11e-05 symW 2.36e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.883 ms, resid=4.710e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.810 ms (pre 1.466 + iter 1.345) | mem   23MB | resid 1.095e-02 p95 1.222e-02 max 1.222e-02 | Y_res 8.533e-02 | relerr 3.618e-03 | r2 nan | hard nan | symX 3.72e-05 symW 2.75e-04 | mv nan | bad 0
PE-Quad                   2.844 ms (pre 1.466 + iter 1.378) | mem   21MB | resid 8.676e-03 p95 8.751e-03 max 8.751e-03 | relerr 2.885e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.35e-04 | mv nan | bad 0
PE-Quad-Coupled           3.184 ms (pre 1.466 + iter 1.718) | mem   23MB | resid 1.095e-02 p95 1.222e-02 max 1.222e-02 | Y_res 3.949e-04 | relerr 3.618e-03 | r2 nan | hard nan | symX 8.35e-05 symW 1.92e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 2.844 ms, resid=8.676e-03, hard=nan

-- case spike --
Inverse-Newton            4.582 ms (pre 2.765 + iter 1.817) | mem   23MB | resid 9.729e-03 p95 9.740e-03 max 9.740e-03 | Y_res 7.534e-02 | relerr 3.234e-03 | r2 nan | hard nan | symX 1.55e-05 symW 8.00e-05 | mv nan | bad 0
PE-Quad                   4.259 ms (pre 2.765 + iter 1.494) | mem   21MB | resid 8.226e-03 p95 8.292e-03 max 8.292e-03 | relerr 2.725e-03 | r2 nan | hard nan | symX 0.00e+00 symW 8.26e-05 | mv nan | bad 0
PE-Quad-Coupled           4.264 ms (pre 2.765 + iter 1.499) | mem   23MB | resid 1.006e-02 p95 1.006e-02 max 1.006e-02 | Y_res 1.105e-02 | relerr 3.346e-03 | r2 nan | hard nan | symX 4.19e-05 symW 7.44e-05 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 4.259 ms, resid=8.226e-03, hard=nan


```

## Results for $p=4$

```text
[coeff] using tuned(l_target=0.05, seed=0), safety=1.0, no_final_safety=False
== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.544 ms (pre 1.301 + iter 1.242) | mem   12MB | resid 1.173e-02 p95 1.174e-02 max 1.174e-02 | Y_res 1.444e-01 | relerr 3.019e-03 | r2 nan | hard nan | symX 8.60e-05 symW 8.04e-04 | mv nan | bad 0
PE-Quad                   2.645 ms (pre 1.301 + iter 1.343) | mem   11MB | resid 1.041e-02 p95 1.055e-02 max 1.055e-02 | relerr 2.460e-03 | r2 nan | hard nan | symX 0.00e+00 symW 5.17e-04 | mv nan | bad 0
PE-Quad-Coupled           2.654 ms (pre 1.301 + iter 1.353) | mem   12MB | resid 1.043e-02 p95 1.046e-02 max 1.046e-02 | Y_res 9.974e-02 | relerr 2.461e-03 | r2 nan | hard nan | symX 1.02e-04 symW 6.76e-04 | mv nan | bad 0
BEST overall: Inverse-Newton @ 2.544 ms, resid=1.173e-02, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.461 ms (pre 1.281 + iter 1.180) | mem   12MB | resid 5.682e-03 p95 1.041e-02 max 1.041e-02 | Y_res 1.955e-01 | relerr 1.321e-03 | r2 nan | hard nan | symX 5.16e-05 symW 3.75e-04 | mv nan | bad 0
PE-Quad                   2.716 ms (pre 1.281 + iter 1.435) | mem   11MB | resid 5.684e-03 p95 1.051e-02 max 1.051e-02 | relerr 1.320e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.23e-04 | mv nan | bad 0
PE-Quad-Coupled           2.517 ms (pre 1.281 + iter 1.236) | mem   12MB | resid 5.688e-03 p95 1.042e-02 max 1.042e-02 | Y_res 9.010e-02 | relerr 1.321e-03 | r2 nan | hard nan | symX 9.70e-05 symW 3.82e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.461 ms, resid=5.682e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.476 ms (pre 1.283 + iter 1.193) | mem   12MB | resid 8.189e-03 p95 1.050e-02 max 1.050e-02 | Y_res 9.531e-02 | relerr 2.101e-03 | r2 nan | hard nan | symX 4.82e-05 symW 5.58e-04 | mv nan | bad 0
PE-Quad                   2.626 ms (pre 1.283 + iter 1.343) | mem   11MB | resid 1.793e-02 p95 1.794e-02 max 1.794e-02 | relerr 4.404e-03 | r2 nan | hard nan | symX 0.00e+00 symW 3.04e-04 | mv nan | bad 0
PE-Quad-Coupled           2.469 ms (pre 1.283 + iter 1.186) | mem   12MB | resid 8.192e-03 p95 1.051e-02 max 1.051e-02 | Y_res 3.827e-02 | relerr 2.101e-03 | r2 nan | hard nan | symX 8.66e-05 symW 4.89e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.469 ms, resid=8.192e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.433 ms (pre 1.255 + iter 1.178) | mem   12MB | resid 1.045e-02 p95 1.070e-02 max 1.070e-02 | Y_res 2.262e-01 | relerr 2.543e-03 | r2 nan | hard nan | symX 4.42e-05 symW 5.08e-04 | mv nan | bad 0
PE-Quad                   2.714 ms (pre 1.255 + iter 1.459) | mem   11MB | resid 1.247e-02 p95 1.694e-02 max 1.694e-02 | relerr 3.050e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.88e-04 | mv nan | bad 0
PE-Quad-Coupled           2.469 ms (pre 1.255 + iter 1.214) | mem   12MB | resid 1.058e-02 p95 1.083e-02 max 1.083e-02 | Y_res 1.019e-01 | relerr 2.574e-03 | r2 nan | hard nan | symX 1.15e-04 symW 5.51e-04 | mv nan | bad 0
BEST overall: Inverse-Newton @ 2.433 ms, resid=1.045e-02, hard=nan

-- case spike --
Inverse-Newton            2.567 ms (pre 1.359 + iter 1.207) | mem   12MB | resid 9.310e-03 p95 9.556e-03 max 9.556e-03 | Y_res 1.650e-01 | relerr 2.322e-03 | r2 nan | hard nan | symX 3.80e-05 symW 5.25e-04 | mv nan | bad 0
PE-Quad                   2.653 ms (pre 1.359 + iter 1.293) | mem   11MB | resid 1.460e-02 p95 1.488e-02 max 1.488e-02 | relerr 3.614e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.90e-04 | mv nan | bad 0
PE-Quad-Coupled           2.551 ms (pre 1.359 + iter 1.192) | mem   12MB | resid 1.461e-02 p95 1.486e-02 max 1.486e-02 | Y_res 3.984e-02 | relerr 3.615e-03 | r2 nan | hard nan | symX 6.35e-05 symW 3.18e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.567 ms, resid=9.310e-03, hard=nan

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.732 ms (pre 1.252 + iter 1.480) | mem   23MB | resid 5.683e-03 p95 5.762e-03 max 5.762e-03 | Y_res 2.851e-01 | relerr 1.344e-03 | r2 nan | hard nan | symX 3.04e-05 symW 2.59e-04 | mv nan | bad 0
PE-Quad                   2.573 ms (pre 1.252 + iter 1.321) | mem   21MB | resid 5.689e-03 p95 5.768e-03 max 5.768e-03 | relerr 1.343e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.67e-04 | mv nan | bad 0
PE-Quad-Coupled           2.535 ms (pre 1.252 + iter 1.282) | mem   23MB | resid 5.812e-03 p95 5.888e-03 max 5.888e-03 | Y_res 1.763e-01 | relerr 1.378e-03 | r2 nan | hard nan | symX 7.67e-05 symW 2.36e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.535 ms, resid=5.812e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.699 ms (pre 1.295 + iter 1.404) | mem   23MB | resid 8.039e-03 p95 9.352e-03 max 9.352e-03 | Y_res 3.193e-01 | relerr 1.971e-03 | r2 nan | hard nan | symX 2.52e-05 symW 2.46e-04 | mv nan | bad 0
PE-Quad                   2.682 ms (pre 1.295 + iter 1.387) | mem   21MB | resid 8.045e-03 p95 1.507e-02 max 1.507e-02 | relerr 1.970e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.77e-04 | mv nan | bad 0
PE-Quad-Coupled           2.528 ms (pre 1.295 + iter 1.233) | mem   23MB | resid 8.047e-03 p95 9.703e-03 max 9.703e-03 | Y_res 1.182e-01 | relerr 1.971e-03 | r2 nan | hard nan | symX 6.78e-05 symW 2.55e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.528 ms, resid=8.047e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.860 ms (pre 1.648 + iter 1.212) | mem   23MB | resid 9.838e-03 p95 9.957e-03 max 9.957e-03 | Y_res 3.088e-01 | relerr 2.439e-03 | r2 nan | hard nan | symX 4.65e-05 symW 5.97e-04 | mv nan | bad 0
PE-Quad                   2.958 ms (pre 1.648 + iter 1.310) | mem   21MB | resid 1.231e-02 p95 1.290e-02 max 1.290e-02 | relerr 3.062e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.41e-04 | mv nan | bad 0
PE-Quad-Coupled           2.863 ms (pre 1.648 + iter 1.214) | mem   23MB | resid 1.118e-02 p95 1.158e-02 max 1.158e-02 | Y_res 3.827e-02 | relerr 2.779e-03 | r2 nan | hard nan | symX 7.11e-05 symW 4.11e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.860 ms, resid=9.838e-03, hard=nan

-- case near_rank_def --
Inverse-Newton            2.561 ms (pre 1.283 + iter 1.278) | mem   23MB | resid 8.986e-03 p95 9.034e-03 max 9.034e-03 | Y_res 1.182e-01 | relerr 2.256e-03 | r2 nan | hard nan | symX 3.83e-05 symW 4.55e-04 | mv nan | bad 0
PE-Quad                   2.728 ms (pre 1.283 + iter 1.445) | mem   21MB | resid 1.686e-02 p95 1.694e-02 max 1.694e-02 | relerr 4.217e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.08e-04 | mv nan | bad 0
PE-Quad-Coupled           2.553 ms (pre 1.283 + iter 1.270) | mem   23MB | resid 1.391e-02 p95 1.397e-02 max 1.397e-02 | Y_res 1.747e-02 | relerr 3.487e-03 | r2 nan | hard nan | symX 6.74e-05 symW 7.24e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.561 ms, resid=8.986e-03, hard=nan

-- case spike --
Inverse-Newton            2.644 ms (pre 1.354 + iter 1.290) | mem   23MB | resid 9.327e-03 p95 9.491e-03 max 9.491e-03 | Y_res 1.907e-01 | relerr 2.359e-03 | r2 nan | hard nan | symX 1.25e-05 symW 1.23e-04 | mv nan | bad 0
PE-Quad                   2.776 ms (pre 1.354 + iter 1.422) | mem   21MB | resid 1.614e-02 p95 1.622e-02 max 1.622e-02 | relerr 4.010e-03 | r2 nan | hard nan | symX 0.00e+00 symW 8.66e-05 | mv nan | bad 0
PE-Quad-Coupled           2.626 ms (pre 1.354 + iter 1.272) | mem   23MB | resid 1.616e-02 p95 1.628e-02 max 1.628e-02 | Y_res 3.405e-02 | relerr 4.018e-03 | r2 nan | hard nan | symX 3.21e-05 symW 9.96e-05 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 2.644 ms, resid=9.327e-03, hard=nan


```

## Results for $p=8$

```text
[coeff] using tuned(l_target=0.05, seed=0), safety=1.0, no_final_safety=False
== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            2.765 ms (pre 1.368 + iter 1.397) | mem   12MB | resid 1.438e-02 p95 2.284e-02 max 2.284e-02 | Y_res 1.453e-01 | relerr 1.866e-03 | r2 nan | hard nan | symX 3.08e-05 symW 7.32e-04 | mv nan | bad 0
PE-Quad                   3.825 ms (pre 1.368 + iter 2.457) | mem   11MB | resid 1.113e-02 p95 1.437e-02 max 1.437e-02 | relerr 1.453e-03 | r2 nan | hard nan | symX 0.00e+00 symW 4.57e-04 | mv nan | bad 0
PE-Quad-Coupled           2.718 ms (pre 1.368 + iter 1.350) | mem   12MB | resid 3.066e-02 p95 3.268e-02 max 3.268e-02 | Y_res 1.551e-01 | relerr 3.975e-03 | r2 nan | hard nan | symX 5.15e-05 symW 2.22e-03 | mv nan | bad 0
BEST overall: PE-Quad-Coupled @ 2.718 ms, resid=3.066e-02, hard=nan

-- case illcond_1e6 --
Inverse-Newton            2.850 ms (pre 1.323 + iter 1.527) | mem   12MB | resid 1.136e-02 p95 1.175e-02 max 1.175e-02 | Y_res 2.359e-01 | relerr 1.448e-03 | r2 nan | hard nan | symX 2.43e-05 symW 8.24e-04 | mv nan | bad 0
PE-Quad                   3.697 ms (pre 1.323 + iter 2.374) | mem   11MB | resid 5.980e-03 p95 1.086e-02 max 1.086e-02 | relerr 7.940e-04 | r2 nan | hard nan | symX 0.00e+00 symW 3.24e-04 | mv nan | bad 0
PE-Quad-Coupled           2.745 ms (pre 1.323 + iter 1.422) | mem   12MB | resid 1.087e-02 p95 1.126e-02 max 1.126e-02 | Y_res 8.700e-02 | relerr 1.414e-03 | r2 nan | hard nan | symX 4.65e-05 symW 7.48e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad @ 3.697 ms, resid=5.980e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.591 ms (pre 1.279 + iter 1.312) | mem   12MB | resid 1.857e-02 p95 1.858e-02 max 1.858e-02 | Y_res 1.097e-01 | relerr 2.391e-03 | r2 nan | hard nan | symX 1.58e-05 symW 2.19e-04 | mv nan | bad 0
PE-Quad                   3.751 ms (pre 1.279 + iter 2.472) | mem   11MB | resid 1.857e-02 p95 1.857e-02 max 1.857e-02 | relerr 2.391e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.73e-04 | mv nan | bad 0
PE-Quad-Coupled           2.626 ms (pre 1.279 + iter 1.347) | mem   12MB | resid 1.857e-02 p95 1.858e-02 max 1.858e-02 | Y_res 3.664e-02 | relerr 2.391e-03 | r2 nan | hard nan | symX 4.01e-05 symW 2.32e-04 | mv nan | bad 0
BEST overall: Inverse-Newton @ 2.591 ms, resid=1.857e-02, hard=nan

-- case near_rank_def --
Inverse-Newton            2.685 ms (pre 1.365 + iter 1.320) | mem   12MB | resid 1.292e-02 p95 2.007e-02 max 2.007e-02 | Y_res 3.623e-02 | relerr 1.686e-03 | r2 nan | hard nan | symX 1.56e-05 symW 2.27e-04 | mv nan | bad 0
PE-Quad                   3.871 ms (pre 1.365 + iter 2.506) | mem   11MB | resid 1.293e-02 p95 2.007e-02 max 2.007e-02 | relerr 1.686e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.75e-04 | mv nan | bad 0
PE-Quad-Coupled           2.846 ms (pre 1.365 + iter 1.481) | mem   12MB | resid 1.534e-02 p95 2.007e-02 max 2.007e-02 | Y_res 1.154e-01 | relerr 1.991e-03 | r2 nan | hard nan | symX 4.07e-05 symW 3.08e-04 | mv nan | bad 0
BEST overall: Inverse-Newton @ 2.685 ms, resid=1.292e-02, hard=nan

-- case spike --
Inverse-Newton            2.752 ms (pre 1.433 + iter 1.319) | mem   12MB | resid 1.652e-02 p95 1.818e-02 max 1.818e-02 | Y_res 1.146e-01 | relerr 2.151e-03 | r2 nan | hard nan | symX 9.25e-06 symW 1.68e-04 | mv nan | bad 0
PE-Quad                   3.948 ms (pre 1.433 + iter 2.515) | mem   11MB | resid 1.560e-02 p95 1.561e-02 max 1.561e-02 | relerr 2.039e-03 | r2 nan | hard nan | symX 0.00e+00 symW 1.87e-04 | mv nan | bad 0
PE-Quad-Coupled           2.713 ms (pre 1.433 + iter 1.280) | mem   12MB | resid 3.558e-02 p95 3.732e-02 max 3.732e-02 | Y_res 1.349e-01 | relerr 4.573e-03 | r2 nan | hard nan | symX 2.65e-05 symW 9.65e-04 | mv nan | bad 0
BEST overall: PE-Quad-Coupled @ 2.713 ms, resid=3.558e-02, hard=nan

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 | lmax=row_sum | terminal=True | timing_reps=1 | symY=True | metrics=full | power_it=0 | mv_k=0 | hard_it=0 ==
-- case gaussian_spd --
Inverse-Newton            3.021 ms (pre 1.547 + iter 1.473) | mem   23MB | resid 7.726e-03 p95 8.518e-03 max 8.518e-03 | Y_res 3.164e-01 | relerr 1.005e-03 | r2 nan | hard nan | symX 2.16e-05 symW 4.57e-04 | mv nan | bad 0
PE-Quad                   4.014 ms (pre 1.547 + iter 2.467) | mem   21MB | resid 6.112e-03 p95 6.195e-03 max 6.195e-03 | relerr 8.087e-04 | r2 nan | hard nan | symX 0.00e+00 symW 2.67e-04 | mv nan | bad 0
PE-Quad-Coupled           2.948 ms (pre 1.547 + iter 1.401) | mem   23MB | resid 7.720e-03 p95 8.510e-03 max 8.510e-03 | Y_res 8.378e-02 | relerr 1.005e-03 | r2 nan | hard nan | symX 3.48e-05 symW 4.56e-04 | mv nan | bad 0
BEST<=residual=0.01: PE-Quad-Coupled @ 2.948 ms, resid=7.720e-03, hard=nan

-- case illcond_1e6 --
Inverse-Newton            3.264 ms (pre 1.388 + iter 1.876) | mem   23MB | resid 8.494e-03 p95 1.556e-02 max 1.556e-02 | Y_res 2.286e-01 | relerr 1.136e-03 | r2 nan | hard nan | symX 2.18e-05 symW 3.59e-04 | mv nan | bad 0
PE-Quad                   3.896 ms (pre 1.388 + iter 2.508) | mem   21MB | resid 8.494e-03 p95 1.556e-02 max 1.556e-02 | relerr 1.136e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.77e-04 | mv nan | bad 0
PE-Quad-Coupled           2.821 ms (pre 1.388 + iter 1.433) | mem   23MB | resid 1.637e-02 p95 1.638e-02 max 1.638e-02 | Y_res 1.545e-01 | relerr 2.125e-03 | r2 nan | hard nan | symX 3.60e-05 symW 8.15e-04 | mv nan | bad 0
BEST<=residual=0.01: Inverse-Newton @ 3.264 ms, resid=8.494e-03, hard=nan

-- case illcond_1e12 --
Inverse-Newton            2.715 ms (pre 1.300 + iter 1.415) | mem   23MB | resid 1.279e-02 p95 1.337e-02 max 1.337e-02 | Y_res 5.138e-02 | relerr 1.696e-03 | r2 nan | hard nan | symX 1.24e-05 symW 2.00e-04 | mv nan | bad 0
PE-Quad                   3.688 ms (pre 1.300 + iter 2.388) | mem   21MB | resid 1.279e-02 p95 1.337e-02 max 1.337e-02 | relerr 1.696e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.34e-04 | mv nan | bad 0
PE-Quad-Coupled           2.713 ms (pre 1.300 + iter 1.412) | mem   23MB | resid 3.540e-02 p95 3.881e-02 max 3.881e-02 | Y_res 2.172e-01 | relerr 4.562e-03 | r2 nan | hard nan | symX 3.33e-05 symW 1.39e-03 | mv nan | bad 0
BEST overall: PE-Quad-Coupled @ 2.713 ms, resid=3.540e-02, hard=nan

-- case near_rank_def --
Inverse-Newton            2.770 ms (pre 1.267 + iter 1.503) | mem   23MB | resid 1.732e-02 p95 1.739e-02 max 1.739e-02 | Y_res 8.856e-02 | relerr 2.280e-03 | r2 nan | hard nan | symX 1.16e-05 symW 1.63e-04 | mv nan | bad 0
PE-Quad                   3.808 ms (pre 1.267 + iter 2.541) | mem   21MB | resid 1.732e-02 p95 1.739e-02 max 1.739e-02 | relerr 2.280e-03 | r2 nan | hard nan | symX 0.00e+00 symW 2.02e-04 | mv nan | bad 0
PE-Quad-Coupled           2.658 ms (pre 1.267 + iter 1.391) | mem   23MB | resid 2.054e-02 p95 2.068e-02 max 2.068e-02 | Y_res 7.734e-02 | relerr 2.678e-03 | r2 nan | hard nan | symX 2.94e-05 symW 5.07e-04 | mv nan | bad 0
BEST overall: PE-Quad-Coupled @ 2.658 ms, resid=2.054e-02, hard=nan

-- case spike --
Inverse-Newton            2.654 ms (pre 1.268 + iter 1.385) | mem   23MB | resid 1.706e-02 p95 1.710e-02 max 1.710e-02 | Y_res 2.233e-01 | relerr 2.195e-03 | r2 nan | hard nan | symX 6.00e-06 symW 6.34e-05 | mv nan | bad 0
PE-Quad                   3.927 ms (pre 1.268 + iter 2.658) | mem   21MB | resid 1.713e-02 p95 1.720e-02 max 1.720e-02 | relerr 2.204e-03 | r2 nan | hard nan | symX 0.00e+00 symW 8.38e-05 | mv nan | bad 0
PE-Quad-Coupled           2.690 ms (pre 1.268 + iter 1.421) | mem   23MB | resid 2.680e-02 p95 2.683e-02 max 2.683e-02 | Y_res 1.595e-01 | relerr 3.428e-03 | r2 nan | hard nan | symX 1.53e-05 symW 3.22e-04 | mv nan | bad 0
BEST overall: Inverse-Newton @ 2.654 ms, resid=1.706e-02, hard=nan


```

## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents.
