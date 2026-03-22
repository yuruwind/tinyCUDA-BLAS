// include/sgemm.h
#pragma once

#include <cstddef> // for size_t
#include <cuda_fp16.h>

// 基础版本 (你现在的 i-k-j)
void sgemm_cpu_naive(int N, float* A, float* B, float* C);

// AVX2 优化版本 (马上要写的)
void sgemm_cpu_avx2(int N, float* A, float* B, float* C);

// Cache Blocking + AVX2 版本 (挑战版)
void sgemm_cpu_block(int N, float* A, float* B, float* C);

// GPU 版本声明
void sgemm_gpu_naive(int N, float* A, float* B, float* C);

void sgemm_gpu_shared(int N, float* A, float* B, float* C);

void sgemm_gpu_cublas(int N, float* A, float* B, float* C);

// 纯计算接口 (输入指针必须指向 GPU 显存)
void sgemm_gpu_naive_device(int N, float* d_A, float* d_B, float* d_C);
void sgemm_gpu_shared_device(int N, float* d_A, float* d_B, float* d_C);
void sgemm_gpu_cublas_device(int N, float* d_A, float* d_B, float* d_C);

void sgemm_gpu_vectorized_device(int N, float* d_A, float* d_B, float* d_C);

//void sgemm_gpu_shared_float4_device(int N, float* d_A, float* d_B, float* d_C);

void sgemm_gpu_2d_tiled_device(int N, float* d_A, float* d_B, float* d_C);

void sgemm_gpu_fused_relu_device(int N, float* d_A, float* d_B, float* d_C);

// 3. 独立的 ReLU
void relu_gpu_standalone_device(int N, float* d_C);

// 增加带 Bias 的版本
void sgemm_gpu_fused_bias_relu_device(int N, float* d_A, float* d_B, float* d_bias, float* d_C);

// Tensor Core 纯计算接口（输入为 half，输出为 float）
void sgemm_gpu_tensor_core_device(int N, const __half* d_A, const __half* d_B, float* d_C);