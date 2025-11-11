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

// Ref
namespace Ref {
    // THIS CODE WAS TAKEN FROM https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
    // WILL BE REPLACED BY MY OWN CODE AFTER ASKING ABOUT PTX INSTRUCTIONS IN OFFICE HOURS

template<int WGMMA_N>
__device__ inline void wgmma_m64nNk16(float d[WGMMA_N/16][8], bf16* sA, bf16* sB) {
    static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 192 || WGMMA_N == 256);
    if  constexpr (WGMMA_N == 256)
        wgmma256<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 192)
        wgmma192<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 128)
        wgmma128<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32<1, 1, 1, 0, 0>(d, sA, sB);
}

template<int WGMMA_N>
__device__ inline void store_wgmma_m64nNk16 (float d[WGMMA_N/16][8], int tid, bf16* block_C, int M) {
    int lane = tid & 31;
    int warp = tid >> 5;
    uint32_t row = warp*16 + lane/4;

    #define CIDX(i,j) (( (j) )*M + ( (i) ))
    #pragma unroll
    for (int w = 0; w < WGMMA_N/16; ++w) {
        int col = 16*w + 2*(tid & 3);

        block_C[CIDX(row,     col    )] = __float2bfloat16(d[w][0]);
        block_C[CIDX(row,     col + 1)] = __float2bfloat16(d[w][1]);
        block_C[CIDX(row + 8, col    )] = __float2bfloat16(d[w][2]);
        block_C[CIDX(row + 8, col + 1)] = __float2bfloat16(d[w][3]);

        block_C[CIDX(row,     col + 8)] = __float2bfloat16(d[w][4]);
        block_C[CIDX(row,     col + 9)] = __float2bfloat16(d[w][5]);
        block_C[CIDX(row + 8, col + 8)] = __float2bfloat16(d[w][6]);
        block_C[CIDX(row + 8, col + 9)] = __float2bfloat16(d[w][7]);
    }
    #undef CIDX
}

}



////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

void get_tensor_map (CUtensorMap* src_map, bf16* src, uint32_t globalRows, uint32_t globalCols, uint32_t sharedRows, uint32_t sharedCols) {
    void* globalAddress = src;
    constexpr uint32_t tensorRank = 2;
    uint64_t globalDim[tensorRank] = {globalCols, globalRows};
    uint64_t globalStrides[tensorRank - 1] = {globalCols * sizeof(bf16)};
    uint32_t boxDim[tensorRank] = {sharedCols, sharedRows};
    uint32_t elementStrides[tensorRank] = {1, 1};
    CUDA_CHECK (cuTensorMapEncodeTiled(
        src_map,                // CUtensorMap *tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        tensorRank,                       // cuuint32_t tensorRank,
        globalAddress,                 // void *globalAddress,
        globalDim,                       // const cuuint64_t *globalDim,
        globalStrides,                     // const cuuint64_t *globalStrides,
        boxDim,                   // const cuuint32_t *boxDim,
        elementStrides,                // const cuuint32_t *elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
}


// TUNABLE
constexpr int TILE_M = 64;
constexpr int TILE_N = 128;
constexpr int TILE_K = 64;
constexpr int WGMMA_N= 128;

// CONSTS
constexpr int WGMMA_M= 64;
constexpr int WGMMA_K= 16;
constexpr int NUM_THREADS = 128;

__global__ void h100_matmul(int M, int N, int K, __grid_constant__ const CUtensorMap A_map, __grid_constant__ const CUtensorMap B_map, bf16 *C) {
    
    __shared__ alignas(8)  uint64_t A_barrier;
    __shared__ alignas(8)  uint64_t B_barrier;

    __shared__ alignas(128) bf16  sA[TILE_M*TILE_K];
    __shared__ alignas(128) bf16  sB[TILE_N*TILE_K];
    float rC [WGMMA_N/16][8] = {0.0f};

    int GlobalI = blockIdx.y * TILE_N;
    int GlobalJ = blockIdx.x * TILE_M;
    int thIdx = threadIdx.x;

    if ( thIdx == 0 ) {
        init_barrier (&A_barrier, NUM_THREADS);
        init_barrier (&B_barrier, NUM_THREADS);
        async_proxy_fence ();
    }
    __syncthreads();


    int A_cur_phase = 0;
    int B_cur_phase = 0;
    for (int GlobalK = 0 ; GlobalK < K ; GlobalK += TILE_K ) {
        // load A    
        if ( thIdx == 0 ) {
            cp_async_bulk_tensor_2d_global_to_shared (
                sA,
                &A_map,
                GlobalK,
                GlobalJ,
                &A_barrier
            );
            expect_bytes_and_arrive (
                &A_barrier,
                TILE_M*TILE_K*sizeof(bf16)
            );

            // load B
            cp_async_bulk_tensor_2d_global_to_shared (
                sB,
                &B_map,
                GlobalK,
                GlobalI,
                &B_barrier
            );
            expect_bytes_and_arrive (
                &B_barrier,
                TILE_N*TILE_K*sizeof(bf16)
            );
        } else {
            arrive (&A_barrier, 1);
            arrive (&B_barrier, 1);
        }

        wait (&A_barrier, A_cur_phase);
        wait (&B_barrier, B_cur_phase); 
        __syncthreads();
        A_cur_phase ^= 1 ;
        B_cur_phase ^= 1 ;

        // Matmul
        warpgroup_arrive();
        #pragma unroll
        for (int LocalK = 0 ; LocalK < TILE_K ; LocalK += WGMMA_K ) {
            Ref::wgmma_m64nNk16<WGMMA_N>(rC, &sA[LocalK], &sB[LocalK]);
        }
        wgmma_commit();
        wgmma_wait<0>();
    }

    // Store 
    Ref::store_wgmma_m64nNk16 <WGMMA_N> (rC, thIdx, C + GlobalI*M + GlobalJ, M);
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {

    // <--- your code here --->
    CUtensorMap A_map{};
    get_tensor_map (&A_map, A, M, K, TILE_M, TILE_K);
    CUtensorMap B_map{};
    get_tensor_map (&B_map, B, N, K, TILE_N, TILE_K);

    dim3 block (NUM_THREADS, 1, 1) ;
    dim3 grid  ((M + TILE_M - 1 )/TILE_M,(N + TILE_N - 1 )/TILE_N, 1) ;
    h100_matmul <<<grid,block>>>(M,N,K,A_map,B_map,C);
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

    const int M = 8192, N = 8192, K = 8192;

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