#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "sgemm.h"

// 定义 GPU 函数指针类型
using gemm_gpu_func = void (*)(int, float*, float*, float*);

// 宏：检查 CUDA 错误
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 专门测试 GPU Kernel 的函数
void run_gpu_benchmark(int N, const std::string& name, gemm_gpu_func func) {
    size_t bytes = N * N * sizeof(float);

    // 1. 准备数据 (在 Host)
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 1.0f);
    std::vector<float> h_C(N * N, 0.0f);

    // 2. 分配显存 (在 Device) - 这部分时间不计入 GFLOPS
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // 3. 搬运数据 - 这部分时间也不计入
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 预热 (Warmup) - 让 GPU 从休眠态唤醒，初始化 Cache
    func(N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 正式计时 (使用 CUDA Event，比 CPU 计时更准)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始
    cudaEventRecord(start);

    // 运行 Kernel (这里只跑一次，如果要更稳可以跑 10 次取平均)
    int iterations = (N <= 1024) ? 10 : 1; // 小矩阵多跑几次
    for(int i=0; i<iterations; i++) {
        func(N, d_A, d_B, d_C);
    }

    // 记录结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 计算平均时间
    double avg_time_ms = milliseconds / iterations;

    // 6. 计算性能
    double ops = 2.0 * N * N * N;
    double gflops = (ops / (avg_time_ms * 1e-3)) / 1e9;

    std::cout << std::left << std::setw(10) << N 
              << std::setw(15) << name
              << std::setw(15) << avg_time_ms 
              << std::setw(15) << gflops << "\n";

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 我们主要关注大矩阵，因为小矩阵 GPU 跑不满
    std::vector<int> sizes = {1024, 2048, 4096}; 

    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "Running GPU Kernel Benchmark (Excluding PCI-e Transfer Time)\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(15) << "Version"
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << "-------------------------------------------------------------------\n";

    for (int N : sizes) {
        run_gpu_benchmark(N, "CUDA Naive", sgemm_gpu_naive_device);
        run_gpu_benchmark(N, "CUDA Shared", sgemm_gpu_shared_device);
        run_gpu_benchmark(N, "cuBLAS", sgemm_gpu_cublas_device);
        std::cout << "-------------------------------------------------------------------\n";
    }

    return 0;
}