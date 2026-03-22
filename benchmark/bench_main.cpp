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


// 在 run_gpu_benchmark 函数中增加对 bias 的处理逻辑
void run_fused_benchmark(int N) {
    size_t matrix_bytes = N * N * sizeof(float);
    size_t bias_bytes = N * sizeof(float);

    // 1. 准备数据
    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 1.0f);
    std::vector<float> h_C(N * N, 0.0f);
    std::vector<float> h_bias(N, 0.5f);

    // 2. 分配显存
    float *d_A, *d_B, *d_C, *d_bias;
    CHECK_CUDA(cudaMalloc(&d_A, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));

    // 3. 拷贝输入
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), bias_bytes, cudaMemcpyHostToDevice));

    // 4. 预热
    sgemm_gpu_fused_bias_relu_device(N, d_A, d_B, d_bias, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. CUDA Event 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iterations = (N <= 1024) ? 10 : 1;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        sgemm_gpu_fused_bias_relu_device(N, d_A, d_B, d_bias, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    double avg_time_ms = milliseconds / iterations;
    double ops = 2.0 * N * N * N;
    double gflops = (ops / (avg_time_ms * 1e-3)) / 1e9;

    std::cout << std::left << std::setw(10) << N
              << std::setw(15) << "Fused G+B+R"
              << std::setw(15) << avg_time_ms
              << std::setw(15) << gflops << "\n";

    // 6. 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

void run_tensor_core_benchmark(int N) {
    size_t float_bytes = N * N * sizeof(float);
    size_t half_bytes = N * N * sizeof(__half);

    // 1. 准备 half 输入和 float 输出
    std::vector<__half> h_A(N * N);
    std::vector<__half> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    // 2. 分配显存
    __half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, half_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, half_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, float_bytes));

    // 3. 拷贝输入
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), half_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), half_bytes, cudaMemcpyHostToDevice));

    // 4. 预热
    sgemm_gpu_tensor_core_device(N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. CUDA Event 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iterations = (N <= 1024) ? 10 : 1;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        sgemm_gpu_tensor_core_device(N, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    double avg_time_ms = milliseconds / iterations;
    double ops = 2.0 * N * N * N;
    double gflops = (ops / (avg_time_ms * 1e-3)) / 1e9;

    std::cout << std::left << std::setw(10) << N
              << std::setw(15) << "Tensor Core"
              << std::setw(15) << avg_time_ms
              << std::setw(15) << gflops << "\n";

    // 6. 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
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
        run_gpu_benchmark(N, "CUDA Float4", sgemm_gpu_vectorized_device);
        run_gpu_benchmark(N, "CUDA 2D-Reg", sgemm_gpu_2d_tiled_device);
        run_tensor_core_benchmark(N);
        run_gpu_benchmark(N, "Fused Gemm+ReLU", sgemm_gpu_fused_relu_device);
        run_fused_benchmark(N);
        std::cout << "-------------------------------------------------------------------\n";
    }


    // ... 之前的 run_gpu_benchmark 函数保持不变 ...

// 在 main 函数的循环里加入以下测试：

for (int N : sizes) {
    std::cout << "--- Size: " << N << " ---\n";

    // 方案 A: 纯 2D-Reg Gemm (Base Line)
    run_gpu_benchmark(N, "Pure Gemm", sgemm_gpu_2d_tiled_device);

    // 方案 B: 融合版 Fused Gemm + ReLU (Optimization Goal)
    run_gpu_benchmark(N, "Fused Gemm+ReLU", sgemm_gpu_fused_relu_device);

    // 方案 C: 非融合版 (手动模拟两个 Kernel 连续调用)
    // 为了测这个，我们需要写一个新的 lambda 或封装函数
    auto unfused_gemm_relu = [](int n, float* da, float* db, float* dc) {
        sgemm_gpu_2d_tiled_device(n, da, db, dc); // 先算 Gemm
        relu_gpu_standalone_device(n, dc);         // 再算 ReLU
    };
    
    // 我们需要修改 run_gpu_benchmark 或者直接手动计时
    // 这里我直接给你一个新的测试逻辑：
    run_gpu_benchmark(N, "Unfused (G+R)", (gemm_gpu_func)unfused_gemm_relu);

    std::cout << "---------------------------------------------------\n";
}

    return 0;
}