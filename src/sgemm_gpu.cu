#include <cstdio>
#include <cuda_runtime.h>
#include "sgemm.h"

// 宏：检查 CUDA 错误 (非常重要，CUDA 报错不像 C++ 那么直接)
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

// -----------------------------------------------------------
// 1. Kernel 函数: 在 GPU 上执行的代码
// __global__ 表示: Host(CPU) 调用，Device(GPU) 执行
// -----------------------------------------------------------
__global__ void sgemm_naive_kernel(int N, float* A, float* B, float* C) {
    // 每一个线程计算 C 的一个元素 C[row][col]
    
    // 计算当前线程的坐标
    // blockIdx: 当前属于哪个方块
    // blockDim: 一个方块有多大
    // threadIdx: 在方块里的编号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------
// 2. Host 函数: 负责分配显存、拷贝数据、启动 Kernel
// -----------------------------------------------------------
void sgemm_gpu_naive(int N, float* A, float* B, float* C) {
    size_t bytes = N * N * sizeof(float);

    // 1. 在 GPU (Device) 上分配内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // 2. 将数据从 CPU (Host) 拷贝到 GPU (Device)
    // 注意: cudaMemcpy 是同步操作，比较慢，会被计入时间
    CHECK_CUDA(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置启动参数
    // 每个 Block 大小为 32x32 (1024个线程，这是上限)
    dim3 threadsPerBlock(32, 32);
    // Grid 大小根据 N 动态计算
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. 启动 Kernel
    sgemm_naive_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);

    // 检查 Kernel 是否启动失败
    CHECK_CUDA(cudaGetLastError());
    
    // 等待 GPU 执行完毕 (因为 Kernel 是异步的)
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 将结果从 GPU 拷回 CPU
    CHECK_CUDA(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 6. 释放显存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}