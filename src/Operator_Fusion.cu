// =========================================================
// Kernel 6: Fused SGEMM + ReLU (2D Register Tiling)
// =========================================================
#include "sgemm.h"
// 设定块大小参数保持不变
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;



// =========================================================
// 独立的 ReLU Kernel (用于对比测试)
// =========================================================
__global__ void relu_standalone_kernel(int size, float* C) {
    // 每一个线程处理一个元素
    int idx = blockIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 从显存读 -> 寄存器算 -> 存回显存
        // 这一步是 Memory Bound 的典型例子
        C[idx] = fmaxf(C[idx], 0.0f);
    }
}

// Host 调用接口
void relu_gpu_standalone_device(int N, float* d_C) {
    int size = N * N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // 简单的 1D 调度
    relu_standalone_kernel<<<blocks, threads>>>(size, d_C);
}




__global__ void sgemm_fused_relu_2d_tiling_kernel(int N, float* A, float* B, float* C) {
    // 1. 线程与块坐标
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int tid = ty * blockDim.x + tx;

    // 2. 声明 Shared Memory
    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN];

    // 3. 声明寄存器 (Register File)
    float threadResults[TM][TN] = {0.0f};

    // 4. 寄存器缓存
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // 5. 加载索引
    const int load_a_row = tid / 2; 
    const int load_a_col = (tid % 2) * 4; 

    const int load_b_row = tid / 32; 
    const int load_b_col = (tid % 32) * 4; 

    // 大循环：遍历 K 维度
    for (int ph = 0; ph < N; ph += BK) {
        // --- 1. 协作加载 Global -> Shared ---
        float4 vecA = reinterpret_cast<float4*>(&A[(by * BM + load_a_row) * N + (ph + load_a_col)])[0];
        As[load_a_col][load_a_row] = vecA.x;
        As[load_a_col+1][load_a_row] = vecA.y;
        As[load_a_col+2][load_a_row] = vecA.z;
        As[load_a_col+3][load_a_row] = vecA.w;

        float4 vecB = reinterpret_cast<float4*>(&B[(ph + load_b_row) * N + (bx * BN + load_b_col)])[0];
        Bs[load_b_row][load_b_col] = vecB.x;
        Bs[load_b_row][load_b_col+1] = vecB.y;
        Bs[load_b_row][load_b_col+2] = vecB.z;
        Bs[load_b_row][load_b_col+3] = vecB.w;

        __syncthreads();

        // --- 2. 核心计算 (寄存器级 GEMM) ---
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                regM[m] = As[k][ty * TM + m];
            }
            for (int n = 0; n < TN; ++n) {
                regN[n] = Bs[k][tx * TN + n];
            }

            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    threadResults[m][n] += regM[m] * regN[n];
                }
            }
        }
        __syncthreads();
    }

    // --- 3. 写回结果 (算子融合点) ---
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
             int c_row = by * BM + ty * TM + m;
             int c_col = bx * BN + tx * TN + n;
             if (c_row < N && c_col < N) {
                 // 提取当前算出的 C 的元素值
                 float final_val = threadResults[m][n];
                 
                 // 💡 融合 ReLU 操作：直接在寄存器上执行
                 // fmaxf 是 CUDA 提供的内置单精度数学函数，底层会映射为单条 FMAX 硬件指令
                 // 这比写 (final_val > 0.0f) ? final_val : 0.0f 更高效，能避免分支预测开销
                 final_val = fmaxf(final_val, 0.0f);
                 
                 // 写入 Global Memory
                 C[c_row * N + c_col] = final_val;
             }
        }
    }
}

// Host 调用
void sgemm_gpu_fused_relu_device(int N, float* d_A, float* d_B, float* d_C) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 128, N / 128);

    sgemm_fused_relu_2d_tiling_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
}



// =========================================================
// Kernel 7: Fused SGEMM + Bias + ReLU (2D Register Tiling)
// =========================================================

__global__ void sgemm_fused_bias_relu_2d_tiling_kernel(int N, float* A, float* B, float* bias, float* C) {
    // 1. 前面的逻辑（Shared Memory 加载、寄存器计算）完全保持不变
    // ... [此处省略与之前相同的 A, B 搬运和 8x8 计算逻辑] ...
    
    // 1. 线程与块坐标
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int tid = ty * blockDim.x + tx;

    // 2. 声明 Shared Memory
    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN];

    // 3. 声明寄存器 (Register File)
    float threadResults[TM][TN] = {0.0f};

    // 4. 寄存器缓存
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // 5. 加载索引
    const int load_a_row = tid / 2; 
    const int load_a_col = (tid % 2) * 4; 

    const int load_b_row = tid / 32; 
    const int load_b_col = (tid % 32) * 4; 

    // 大循环：遍历 K 维度
    for (int ph = 0; ph < N; ph += BK) {
        // --- 1. 协作加载 Global -> Shared ---
        float4 vecA = reinterpret_cast<float4*>(&A[(by * BM + load_a_row) * N + (ph + load_a_col)])[0];
        As[load_a_col][load_a_row] = vecA.x;
        As[load_a_col+1][load_a_row] = vecA.y;
        As[load_a_col+2][load_a_row] = vecA.z;
        As[load_a_col+3][load_a_row] = vecA.w;

        float4 vecB = reinterpret_cast<float4*>(&B[(ph + load_b_row) * N + (bx * BN + load_b_col)])[0];
        Bs[load_b_row][load_b_col] = vecB.x;
        Bs[load_b_row][load_b_col+1] = vecB.y;
        Bs[load_b_row][load_b_col+2] = vecB.z;
        Bs[load_b_row][load_b_col+3] = vecB.w;

        __syncthreads();

        // --- 2. 核心计算 (寄存器级 GEMM) ---
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                regM[m] = As[k][ty * TM + m];
            }
            for (int n = 0; n < TN; ++n) {
                regN[n] = Bs[k][tx * TN + n];
            }

            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    threadResults[m][n] += regM[m] * regN[n];
                }
            }
        }
        __syncthreads();
    }
    
    // 我们直接跳到最后的写回阶段：
    
    // --- 3. 写回结果 (Epilogue Fusion) ---
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
             int c_row = by * BM + ty * TM + m;
             int c_col = bx * BN + tx * TN + n;
             
             if (c_row < N && c_col < N) {
                 // 1. 获取累加的乘法结果
                 float val = threadResults[m][n];
                 
                 // 2. 融合 Bias：从显存读取该行对应的偏置
                 // 💡 这里的访问非常高效，因为同一行的 8 个结果共享同一个 bias 值
                 val += bias[c_row];
                 
                 // 3. 融合 ReLU
                 val = fmaxf(val, 0.0f);
                 
                 // 4. 一次性写回
                 C[c_row * N + c_col] = val;
             }
        }
    }
}

// Host 调用接口
void sgemm_gpu_fused_bias_relu_device(int N, float* d_A, float* d_B, float* d_bias, float* d_C) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 128, N / 128);

    sgemm_fused_bias_relu_2d_tiling_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_bias, d_C);
}