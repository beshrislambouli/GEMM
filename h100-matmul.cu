// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh", "kernel.cu"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>
#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

// for (int i = 0 ; i < N ; i ++ ) {
//     for (int j = 0 ; j < M ; j ++ ) {
//         float sum = 0.0f;
//         for (int k = 0 ; k < K ; k ++ ) {
//             float a = __bfloat162float (A [IDX(j,k,K)]) ;
//             float b = __bfloat162float (B [IDX(i,k,K)]) ;
//             sum += a * b ; 
//         }
//         C [IDX(i,j,M)] = __float2bfloat16 (sum) ; 
//     }
// }

// ROW MAJOR
#define IDX(i, j, cols) ((i) * (cols) + (j))
constexpr int TILE_N = 64;
constexpr int TILE_M = 64;
constexpr int TILE_K = 64;

__global__ void h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    __shared__ alignas(128) bf16  sA[TILE_M][TILE_K];
    __shared__ alignas(128) bf16  sB[TILE_N][TILE_K];
    __shared__ alignas(128) float sC[TILE_N][TILE_M];

    int GlobalI = blockIdx.y * TILE_N;
    int GlobalJ = blockIdx.x * TILE_M;
    
    for (int i = 0 ; i < TILE_N ; i ++ ) {
        for (int j = 0 ; j < TILE_M ; j ++ ) {
            sC [i][j] = __bfloat162float (0.0f);
        }
    }

    for (int GlobalK = 0 ; GlobalK < K ; GlobalK += TILE_K ) {
        // load A
        for (int i = 0 ; i < TILE_M ; i ++ ) {
            for (int j = 0 ; j < TILE_K ; j ++ ) {
                int gI = GlobalJ + i ;
                int gJ = GlobalK + j ;
                sA [i][j] = A [IDX(gI,gJ,K)];
            }
        }        

        // load B
        for (int i = 0 ; i < TILE_N ; i ++ ) {
            for (int j = 0 ; j < TILE_K ; j ++ ) {
                int gI = GlobalI + i ;
                int gJ = GlobalK + j ;
                sB [i][j] = B [IDX(gI,gJ,K)];
            }
        }

        __syncthreads();

        // Matmul
        
        for (int i = 0 ; i < TILE_N ; i ++ ) {
            for (int j = 0 ; j < TILE_M ; j ++ ) {
                float sum = 0.0f;
                for (int k = 0 ; k < TILE_K ; k ++ ) {
                    float a = __bfloat162float (sA [j][k]);
                    float b = __bfloat162float (sB [i][k]);
                    sum += a * b ;
                }
                sC [i][j] += sum;
            }
        }

        __syncthreads();
    }

    // Store 
    for (int i = 0 ; i < TILE_N ; i ++ ) {
        for (int j = 0 ; j < TILE_M ; j ++ ) {
            int gI = GlobalI + i ;
            int gJ = GlobalJ + j ;
            C [IDX(gI,gJ,M)] = __float2bfloat16 (sC [i][j]);
        }
    }
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {

    // <--- your code here --->
    dim3 block (1, 1, 1) ;
    dim3 grid  ((M + TILE_M - 1 )/TILE_M,(N + TILE_N - 1 )/TILE_N, 1) ;
    h100_matmul <<<grid,block>>>(M,N,K,A,B,C);
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;


#define BENCHPRESS(func, flops, ...)                                           \
    do {                                                                       \
        std::cout << "Running " << #func << " ...\n";                          \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i) {                  \
            func(__VA_ARGS__);                                                 \
        }                                                                      \
        cudaDeviceSynchronize();                                               \
        std::vector<float> times(kNumOfOuterIterations);                       \
        cudaEvent_t start, stop;                                               \
        cudaEventCreate(&start);                                               \
        cudaEventCreate(&stop);                                                \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) {                   \
            cudaEventRecord(start);                                            \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) {               \
                func(__VA_ARGS__);                                             \
            }                                                                  \
            cudaEventRecord(stop);                                             \
            cudaEventSynchronize(stop);                                        \
            float elapsed_time;                                                \
            cudaEventElapsedTime(&elapsed_time, start, stop);                  \
            times[i] = elapsed_time / kNumOfInnerIterations;                   \
        }                                                                      \
        cudaEventDestroy(start);                                               \
        cudaEventDestroy(stop);                                                \
        std::sort(times.begin(), times.end());                                 \
        float best_time_ms = times[0];                                         \
        float tflops = (flops * 1e-9) / best_time_ms;                          \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl;      \
        std::cout << "  TFLOP/s: " << tflops << std::endl;                     \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, N, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N) {
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f) {
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++) {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance) {
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff
                          << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main() {

    const int M = 4096, N = 4096, K = 4096;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}