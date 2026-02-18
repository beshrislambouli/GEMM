# H100 BF16 GEMM (TMA + WGMMA) 

This repo contains a custom **BF16 matrix multiplication** kernel optimized for **NVIDIA H100** using:
- **TMA (Tensor Memory Accelerator)** for async 2D global↔shared transfers via `CUtensorMap`
- **WGMMA** (warp-group MMA) for high-throughput tensor core compute
- A **producer/consumer pipeline** with a **3-stage shared-memory queue** to overlap loads and compute

Target problem size:
- **M = N = K = 8192** (square GEMM)

---

## What it does

Computes `C = Aᵀ × B` in BF16 (with FP32 accumulation inside the WGMMA path), and compares output against a cuBLAS BF16 GEMM reference. Then it benchmarks both implementations using CUDA events.

---

## Performance (H100)

Measured results for `8192 × 8192 × 8192`:

- **cuBLAS (runCublasRef)**  
  Runtime: **1.40367 ms**  
  Throughput: **783.31 TFLOP/s**

- **This kernel (launch_h100_matmul)**  
  Runtime: **1.47893 ms**  
  Throughput: **743.449 TFLOP/s**

Relative performance: **~95% of cuBLAS** 

---
