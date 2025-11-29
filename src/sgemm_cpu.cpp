// src/sgemm_cpu.cpp
#include "sgemm.h"
#include <immintrin.h> // AVX2 的头文件，接下来要用
#define BLOCK_SIZE 32

__attribute__((optimize("no-tree-vectorize")))
void sgemm_cpu_naive(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_val = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

// ⬇️ 你的下一个挑战：实现这个函数
void sgemm_cpu_avx2(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // 1. 将 A[i*N+k] 广播到寄存器 (因为在内层循环它是不变的)
            __m256 vec_a = _mm256_set1_ps(A[i * N + k]);

            // 2. 内层循环每次步进 8
            for (int j = 0; j < N; j += 8) {
                // 加载 C 的当前 8 个值
                __m256 vec_c = _mm256_loadu_ps(&C[i * N + j]);
                
                // 加载 B 的当前 8 个值
                __m256 vec_b = _mm256_loadu_ps(&B[k * N + j]);

                // 计算 C += A * B
                // fmadd(a, b, c) 相当于 a * b + c
                vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);

                // 存回 C
                _mm256_storeu_ps(&C[i * N + j], vec_c);
            }
        }
    }
}

// **【Cache Blocking + AVX2 版本】**
void sgemm_cpu_block(int N, float* A, float* B, float* C) {
    // 外层循环：遍历块 (Block Loop)
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                
                // 内层循环：在块内部进行 AVX2 计算 (Micro Kernel)
                // 这里的逻辑和你之前的 sgemm_cpu_avx2 几乎一样，
                // 只是边界变成了 min(bi + BLOCK_SIZE, N)
                
                for (int i = bi; i < bi + BLOCK_SIZE; ++i) {
                    for (int k = bk; k < bk + BLOCK_SIZE; ++k) {
                        
                        __m256 vec_a = _mm256_set1_ps(A[i * N + k]);
                        
                        for (int j = bj; j < bj + BLOCK_SIZE; j += 8) {
                            __m256 vec_c = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 vec_b = _mm256_loadu_ps(&B[k * N + j]);
                            
                            vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
                            
                            _mm256_storeu_ps(&C[i * N + j], vec_c);
                        }
                    }
                }
                
            }
        }
    }
}