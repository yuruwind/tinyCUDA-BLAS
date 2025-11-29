// include/sgemm.h
#pragma once

#include <cstddef> // for size_t

// 基础版本 (你现在的 i-k-j)
void sgemm_cpu_naive(int N, float* A, float* B, float* C);

// AVX2 优化版本 (马上要写的)
void sgemm_cpu_avx2(int N, float* A, float* B, float* C);

// Cache Blocking + AVX2 版本 (挑战版)
void sgemm_cpu_block(int N, float* A, float* B, float* C);