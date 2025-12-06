#include <cublas_v2.h> // 放在最上面
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

// 定义 Block 大小，必须和 Host 端设置的一样
#define BLOCK_SIZE 16

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


// -----------------------------------------------------------
// 2. Shared Memory Tiling Kernel
// -----------------------------------------------------------
__global__ void sgemm_shared_mem_kernel(int N, float* A, float* B, float* C) {
    // blockIdx: 当前 Block 的坐标
    // threadIdx: 当前线程在 Block 内的坐标
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算当前线程负责计算 C 中的哪个元素坐标
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // 声明 Shared Memory (这是 Block 内所有线程共享的)
    // 大小是 32x32 的 float 矩阵
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0.0f;

    // 核心循环：将大矩阵拆分成一个个 BLOCK_SIZE 宽度的“条” (Tile) 来遍历
    // ph (Phase) 代表当前处理第几个 Tile
    for (int ph = 0; ph < N / BLOCK_SIZE; ++ph) {
        
        // --- 1. 协作加载数据到 Shared Memory ---
        
        // 每个线程负责搬运 A 矩阵的一个点：A[row][ph * BLOCK_SIZE + tx]
        As[ty][tx] = A[row * N + ph * BLOCK_SIZE + tx];
        
        // 每个线程负责搬运 B 矩阵的一个点：B[ph * BLOCK_SIZE + ty][col]
        Bs[ty][tx] = B[(ph * BLOCK_SIZE + ty) * N + col];

        // 🚧 线程同步栅栏 (必考点!) 🚧
        // 必须等待 Block 内所有线程都把数据搬完了，才能开始计算
        __syncthreads();

        // --- 2. 在 Shared Memory 上进行计算 ---
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // 现在是从高速的 Shared Memory (As, Bs) 取数，而不是慢速的 A, B
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // 🚧 再次同步 🚧
        // 必须等待所有线程都算完了当前这个 Tile，才能进入下一轮循环去覆盖 As, Bs
        __syncthreads();
    }

    // 写回结果
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// 对应的 Host 调用函数
void sgemm_gpu_shared(int N, float* A, float* B, float* C) {
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 调用新的 Kernel
    sgemm_shared_mem_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}


// 3. CuBLAS 版本 (官方闭源库)
void sgemm_gpu_cublas(int N, float* A, float* B, float* C) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 1.0f; // 注意这里 beta=1，意味着 C += A*B，符合我们的 benchmark 逻辑
    // 如果想要纯净的 C = A*B，beta 应该设为 0，且外部 benchmark 需要清零 C

    // 关键点：CuBLAS 默认是列主序 (Column Major)，而我们是行主序 (Row Major)。
    // C = A * B (Row Major) 等价于 C^T = B^T * A^T (Column Major)
    // 所以这里我们需要“骗”一下 CuBLAS：
    // 传进去 B 当作 A，传进去 A 当作 B，最后算出来的结果直接就是行主序的 C。
    
    // 解释参数：
    // Handle, OP_N (不转置), OP_N (不转置), 
    // M=N, N=N, K=N (矩阵大小),
    // alpha, 
    // B (作为第一个矩阵), ldb=N, 
    // A (作为第二个矩阵), lda=N, 
    // beta, 
    // C (结果), ldc=N
    
    // 显存指针
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    // C 这里如果是累加，需要把 Host 的 C 拷进去；如果是覆盖，则不需要。
    // 为了公平，假设是覆盖 (beta=0) 或累加。这里简单起见，我们假设是纯计算。
    // 修正：benchmark loop 里我们通常把 C 设为 0，所以这里 beta=0 比较合适。
    float beta_overwrite = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_B, N, // B 换到前面
                d_A, N, // A 换到后面
                &beta_overwrite, 
                d_C, N);

    cudaDeviceSynchronize(); // 等待计算完成

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}


// =========================================================
// 新增：纯计算接口 (Device Pointers Only)
// 这些函数假设 d_A, d_B, d_C 已经在显存里了，只负责计算
// =========================================================

// 1. Naive Device 接口
void sgemm_gpu_naive_device(int N, float* d_A, float* d_B, float* d_C) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sgemm_naive_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
}

// 2. Shared Memory Device 接口 (Block Size = 16)
void sgemm_gpu_shared_device(int N, float* d_A, float* d_B, float* d_C) {
    // 强制使用 Block Size 16 (配合之前的宏定义)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    sgemm_shared_mem_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
}

// 3. CuBLAS Device 接口
void sgemm_gpu_cublas_device(int N, float* d_A, float* d_B, float* d_C) {
    // 为了性能，handle 应该在外部创建，但这里为了接口简单先放在这
    // 注意：频繁创建 handle 也有开销，但在大矩阵下可以忽略
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f; 

    // 再次提醒：CuBLAS 是列主序，我们交换 A/B 来欺骗它
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_B, N, 
                d_A, N, 
                &beta, 
                d_C, N);
    
    // 这里不需要 DeviceSynchronize，因为 benchmark 主程序会做
    cublasDestroy(handle);
}