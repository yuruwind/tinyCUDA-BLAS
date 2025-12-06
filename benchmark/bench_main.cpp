#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath> // For std::abs
#include <algorithm> // For std::fill

// 引入我们的库头文件
#include "sgemm.h" 

// 定义一个函数指针类型，用于指向不同的GEMM实现
using gemm_func = void (*)(int, float*, float*, float*);

/**
 * 运行并测试一个GEMM版本
 */
void run_benchmark_for_version(
    int N, 
    float* A, 
    float* B, 
    float* C, 
    const std::string& version_name, 
    gemm_func func
) {
    // ⚠️ 每次运行前必须清零 C，因为 i-k-j 顺序是累加操作！
    // 假设 C 的大小是 N*N
    std::fill(C, C + N * N, 0.0f); 

    auto start = std::chrono::high_resolution_clock::now();
    func(N, A, B, C); // 调用传入的函数
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;

    // FLOPS = 2 * N^3
    double ops = 2.0 * N * N * N;
    double gflops = (ops / (duration.count() * 1e-3)) / 1e9;

    // 打印结果
    std::cout << std::left << std::setw(10) << N 
              << std::setw(15) << version_name
              << std::setw(15) << duration.count() 
              << std::setw(15) << gflops << "\n";
}

int main() {
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    
    // ------------------- 表头 ---------------------
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running AVX2 Optimization Benchmark\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(15) << "Version"
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << "-------------------------------------------------------------------\n";

    for (int N : sizes) {
        // 1. 准备数据 (A, B, C 只需要准备一次)
        std::vector<float> A(N * N, 1.0f);
        std::vector<float> B(N * N, 1.0f);
        std::vector<float> C(N * N, 0.0f); // C 每次运行前都会被 run_benchmark_for_version 清零

        // 2. 运行 Reordered Naive (i-k-j)
        run_benchmark_for_version(N, A.data(), B.data(), C.data(), "Reordered", sgemm_cpu_naive);

        // 3. 运行 AVX2 Optimized
        run_benchmark_for_version(N, A.data(), B.data(), C.data(), "AVX2", sgemm_cpu_avx2);

        //run_benchmark_for_version(N, A.data(), B.data(), C.data(), "Blocked AVX2", sgemm_cpu_block);

        // 4. 运行 CUDA Naive
        // 注意：第一次运行 CUDA 可能稍慢（初始化 Context），是正常的
        run_benchmark_for_version(N, A.data(), B.data(), C.data(), "CUDA Naive", sgemm_gpu_naive);
    }

    return 0;
}