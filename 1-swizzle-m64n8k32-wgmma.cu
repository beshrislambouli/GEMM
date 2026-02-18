// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: 64B Swizzle WGGMA load for M = 64, N = 8, K = 32
////////////////////////////////////////////////////////////////////////////////

__device__ int f(int row, int col) {
    int group = (row & 7) >> 1;
    int block  = col >> 3;
    int offset = col & 7;
    int delta = (block & 1) ? -group : group;
    int new_block = (block + delta) & 3;

    return (new_block << 3) | offset;
}

template <int TILE_M, int TILE_K>
__device__ void load_a(bf16* gmem, bf16* smem) {
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
        int row = idx / TILE_K;
        int col = idx - row * TILE_K;
        int swizzled_col = f(row, col);
        int g_idx = row * TILE_K + col;
        int s_idx = row * TILE_K + swizzled_col;

        smem[s_idx] = gmem[g_idx];
    }
}

template <int TILE_N, int TILE_K>
__device__ void load_b(bf16* gmem, bf16* smem) {
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < TILE_N * TILE_K; idx += blockDim.x * blockDim.y) {
        int row = idx / TILE_K;
        int col = idx - row * TILE_K;
        int swizzled_col = f(row, col);
        int g_idx = row * TILE_K + col;
        int s_idx = row * TILE_K + swizzled_col;

        smem[s_idx] = gmem[g_idx];
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {

    __shared__ bf16 sA[TILE_M * TILE_K];
    __shared__ bf16 sB[TILE_N * TILE_K];
    float d[4] = {0.f, 0.f, 0.f, 0.f};

    load_a<TILE_M, TILE_K>(a, sA);
    load_b<TILE_N, TILE_K>(b, sB);

    __syncthreads();
    async_proxy_fence();

    warpgroup_arrive();
    wgmma_n8<0, 1, 1, 0, 0>(
        make_smem_desc<SWIZZLE_64B>(sA, 1, 512),
        make_smem_desc<SWIZZLE_64B>(sB, 1, 512),
        d
    );
    wgmma_n8<1, 1, 1, 0, 0>(
        make_smem_desc<SWIZZLE_64B>(sA + 16, 1, 512),
        make_smem_desc<SWIZZLE_64B>(sB + 16, 1, 512),
        d
    );
    wgmma_commit();
    wgmma_wait<0>();

    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int row0 = (warp * 16) + (lane >> 2);
    int row1 = row0 + 8;
    int col0 = (lane & 3) << 1;
    int col1 = col0 + 1;
    c[row0 + TILE_M * col0] = d[0];
    c[row0 + TILE_M * col1] = d[1];
    c[row1 + TILE_M * col0] = d[2];
    c[row1 + TILE_M * col1] = d[3];
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {
    dim3 block(32, 4, 1);
    dim3 grid(1, 1, 1);
    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<grid, block>>>(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 32;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[j * N + i] = (i + j) / 10.0f;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning Swizzle WGMMA M=64, N=8, K-32...\n\n");
    launch_swizzle_wgmma_m64n8k32<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++) {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f) {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}