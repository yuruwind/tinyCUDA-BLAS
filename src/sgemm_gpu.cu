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


// =========================================================
// Kernel 3: Vectorized Memory Access (float4)
// =========================================================

// 强制将 float* 转换为 float4* 读取
__global__ void sgemm_vectorized_kernel(int N, float* A, float* B, float* C) {
    // 这里的 x 代表“向量”的坐标，每个 x 处理 4 个 float
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col 是向量索引

    // 实际的列坐标需要 x4
    int actual_col = col * 4;

    if (row < N && actual_col < N) {
        // Cvalue 用 float4 来存，一次算 4 个结果
        float4 c_res = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = 0; k < N; ++k) {
            float a_val = A[row * N + k]; // A 还是一个一个读

            // B 一次读 4 个！ (关键优化)
            // 我们要求 N 是 4 的倍数，且 B 的地址对齐
            float4 b_val = reinterpret_cast<float4*>(&B[k * N + actual_col])[0];

            // 手动展开计算 4 个点
            c_res.x += a_val * b_val.x;
            c_res.y += a_val * b_val.y;
            c_res.z += a_val * b_val.z;
            c_res.w += a_val * b_val.w;
        }

        // 结果一次性写回 4 个
        reinterpret_cast<float4*>(&C[row * N + actual_col])[0] = c_res;
    }
}

// Host 函数
void sgemm_gpu_vectorized_device(int N, float* d_A, float* d_B, float* d_C) {
    // Block 还是 32x32，但 x 维度只需要原来的 1/4
    dim3 threadsPerBlock(32 / 4, 32); 
    dim3 numBlocks((N / 4 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    sgemm_vectorized_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
}


// =========================================================
// Kernel 5: 2D Register Tiling (终极优化)
// =========================================================

// 设定块大小参数
// BM, BN: 一个 Block 计算 C 的 128x128 区域
// BK: K 维度每次切分 8
// TM, TN: 每个线程计算 C 的 8x8 区域
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_2d_register_tiling_kernel(int N, float* A, float* B, float* C) {
    // 1. 线程与块坐标
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 线程在 Block 内的线性 ID (0 ~ 255)
    const int tid = ty * blockDim.x + tx;

    // 2. 声明 Shared Memory
    // As: [2][BK][BM] -> 使用双缓冲思路防止 Bank Conflict (这里简化为单缓冲但转置存储)
    // 为了避免 Bank Conflict，我们通常会把 shared memory 设大一点或者转置
    // 这里采用简单方案：As[BK][BM], Bs[BK][BN]
    // 实际上对于 BM=128，需要很大的 Shared Mem。
    // 注意：4060 的 Shared Memory 足够大 (48KB/Block 以上)。
    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN];

    // 3. 声明寄存器 (Register File)
    // 每个线程负责 8x8 的累加结果，这 64 个 float 必须完全驻留在寄存器中
    float threadResults[TM][TN] = {0.0f};

    // 4. 寄存器缓存，用于从 Shared Mem 读取数据
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // 5. 计算加载 Global Memory 的索引
    // 我们需要由 256 个线程 (16x16) 协作搬运 A (128x8) 和 B (8x128)
    
    // A_row, A_col: 当前线程负责搬运 A 的哪个点
    // A 是 128行 x 8列。总共 1024 个元素。
    // 线程数 256。每个线程搬运 4 个 float。
    // 使用 float4 搬运！
    const int load_a_row = tid / 2; // 0~127
    const int load_a_col = (tid % 2) * 4; // 0, 4

    // B_row, B_col: 当前线程负责搬运 B 的哪个点
    // B 是 8行 x 128列。总共 1024 个元素。
    // 每个线程搬运 4 个。
    const int load_b_row = tid / 32; // 0~7
    const int load_b_col = (tid % 32) * 4; // 0, 4, ..., 124

    // 大循环：遍历 K 维度
    for (int ph = 0; ph < N; ph += BK) {
        // --- 1. 协作加载 Global -> Shared ---
        
        // 加载 A (转置存入 Shared Mem 以优化读取) -> As[col][row]
        // 使用 float4 向量化加载
        float4 vecA = reinterpret_cast<float4*>(&A[(by * BM + load_a_row) * N + (ph + load_a_col)])[0];
        As[load_a_col][load_a_row] = vecA.x;
        As[load_a_col+1][load_a_row] = vecA.y;
        As[load_a_col+2][load_a_row] = vecA.z;
        As[load_a_col+3][load_a_row] = vecA.w;

        // 加载 B -> Bs[row][col]
        float4 vecB = reinterpret_cast<float4*>(&B[(ph + load_b_row) * N + (bx * BN + load_b_col)])[0];
        Bs[load_b_row][load_b_col] = vecB.x;
        Bs[load_b_row][load_b_col+1] = vecB.y;
        Bs[load_b_row][load_b_col+2] = vecB.z;
        Bs[load_b_row][load_b_col+3] = vecB.w;

        __syncthreads();

        // --- 2. 核心计算 (寄存器级 GEMM) ---
        // 外层循环：在 Shared Memory 的 BK 维度上迭代 (0~7)
        for (int k = 0; k < BK; ++k) {
            // 将 Shared Memory 的数据预加载到寄存器
            // 这一步极大减少了 Shared Memory 的访问压力
            for (int m = 0; m < TM; ++m) {
                regM[m] = As[k][ty * TM + m];
            }
            for (int n = 0; n < TN; ++n) {
                regN[n] = Bs[k][tx * TN + n];
            }

            // 外积计算 (Outer Product)
            // 8x8 = 64 次乘加，纯寄存器操作，极快！
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    threadResults[m][n] += regM[m] * regN[n];
                }
            }
        }

        __syncthreads();
    }

    // --- 3. 写回结果 ---
    // 每个线程负责写回 C 的 8x8 区域
    // 这一步不需要极致优化，因为它在 Kernel 结束时只执行一次
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
             int c_row = by * BM + ty * TM + m;
             int c_col = bx * BN + tx * TN + n;
             if (c_row < N && c_col < N) {
                 C[c_row * N + c_col] = threadResults[m][n];
             }
        }
    }
}

// Host 调用
void sgemm_gpu_2d_tiled_device(int N, float* d_A, float* d_B, float* d_C) {
    // 这里的 BlockSize 是线程数的维度
    // 我们用 16x16 = 256 个线程
    // 每个线程算 8x8，所以一个 Block 算 (16*8) x (16*8) = 128x128
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 128, N / 128); // 假设 N 是 128 的倍数

    sgemm_2d_register_tiling_kernel<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);
}