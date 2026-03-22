#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "sgemm.h"

using namespace nvcuda;

#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

// Tensor Core 固定的 Tile 尺寸
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// 每个 block 内的 warp 布局：2x2 个 warp，共 4 个 warp。
const int WARPS_M = 2;
const int WARPS_N = 2;


// =========================================================
// 优化版 Tensor Core Kernel (Shared Memory + Warp Tiling)
// =========================================================
__global__ void sgemm_tensor_core_smem_kernel(int N, const __half* A, const __half* B, float* C) {
    // 1. 定义 Shared Memory (Block 大小：64x64，K 步长：16)
    __shared__ __half As[64][16 + 8];
    __shared__ __half Bs[16][64 + 8]; // +8 是为了避免 bank conflict，实际访问时按 16 步长跳过

    // 2. 声明 Fragment 数组
    // 一个 Warp 负责 32x32 的区域，需要 2x2 个 16x16 的累加器
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[2][2];

    // 初始化累加器
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    // 3. 计算坐标
    // Block 有 128 个线程 (4个Warp)。Warp 排列为 2x2。
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2; // 0 或 1
    int warp_col = warp_id % 2; // 0 或 1

    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 64;

    // 4. 主循环 (沿 K 维度)
    for (int k = 0; k < N; k += 16) {
        
        // --- 第一步：128 个线程协作，将数据从 Global 搬到 Shared ---
        // As 需搬运 64x16=1024 个元素；Bs 需搬运 16x64=1024 个元素
        // 每个线程搬运 8 个元素 (1024 / 128)
        for (int step = 0; step < 1024; step += 128) {
            int idx = step + threadIdx.x;
            if (idx < 1024) {
                // 搬运 A
                As[idx / 16][idx % 16] = A[(block_row + idx / 16) * N + k + idx % 16];
                // 搬运 B
                Bs[idx / 64][idx % 64] = B[(k + idx / 64) * N + block_col + idx % 64];
            }
        }
        __syncthreads(); // 等待所有人搬完

        // --- 第二步：从 Shared Memory 加载到 Fragment ---
        for (int i = 0; i < 2; i++) {
            // 注意 Shared Memory 的 leading dimension
            wmma::load_matrix_sync(a_frag[i], &As[warp_row * 32 + i * 16][0], 16);
            wmma::load_matrix_sync(b_frag[i], &Bs[0][warp_col * 32 + i * 16], 64);
        }

        // --- 第三步：疯狂发射 MMA 指令 ---
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
            }
        }
        __syncthreads(); // 准备下一轮搬运
    }

    // 5. 写回 Global Memory
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int c_row = block_row + warp_row * 32 + i * 16;
            int c_col = block_col + warp_col * 32 + j * 16;
            if (c_row < N && c_col < N) {
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Host 调用需要更新 Block 和 Grid 大小
void sgemm_gpu_tensor_core_device(int N, const __half* d_A, const __half* d_B, float* d_C) {
    // 4 个 Warp = 128 线程。每个 Block 处理 64x64
    dim3 threadsPerBlock(128);
    dim3 numBlocks(N / 64, N / 64);

    sgemm_tensor_core_smem_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
    CHECK_CUDA(cudaGetLastError());
}