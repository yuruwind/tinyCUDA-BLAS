#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// 声明外部函数 (将来在 src 中实现)
void naive_gemm(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // 测试矩阵大小
    std::vector<int> sizes = {64, 128, 256, 512, 1024};

    std::cout << "----------------------------------------------------\n";
    std::cout << "Running Manual Benchmark (TinyCUDA-BLAS)\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << "----------------------------------------------------\n";

    for (int N : sizes) {
        // 1. 准备数据
        std::vector<float> A(N * N, 1.0f);
        std::vector<float> B(N * N, 1.0f);
        std::vector<float> C(N * N, 0.0f);

        // 2. 计时开始
        auto start = std::chrono::high_resolution_clock::now();

        // 3. 运行内核 (现在是空的 Naive 实现)
        naive_gemm(N, A.data(), B.data(), C.data());

        // 4. 计时结束
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // 5. 计算性能
        // FLOPS = 2 * N^3 (乘法加法各一次)
        double ops = 2.0 * N * N * N;
        double gflops = (ops / (duration.count() * 1e-3)) / 1e9;

        std::cout << std::left << std::setw(10) << N 
                  << std::setw(15) << duration.count() 
                  << std::setw(15) << gflops << "\n";
    }

    return 0;
}
