cmake_minimum_required(VERSION 3.20)

project(TinyCUDA-BLAS LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 性能优化选项
# 你的 4060 是 Ada Lovelace 架构，对应 sm_89
set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "-O3 -arch=sm_89") 

# --- 修复点：显式查找 OpenMP ---
find_package(OpenMP REQUIRED)

# 寻找 OpenBLAS
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

find_package(CUDAToolkit REQUIRED) # 确保这一行在(openmp)

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 1. 核心库
add_library(tinycuda_blas 
    src/sgemm_cpu.cpp 
    src/sgemm_gpu.cu
)
# 链接 OpenMP

target_link_libraries(tinycuda_blas PRIVATE OpenMP::OpenMP_CXX CUDA::cublas)

# 2. Benchmark 可执行文件
add_executable(run_bench benchmark/bench_main.cpp)
# 链接核心库和 OpenBLAS
target_link_libraries(run_bench PRIVATE tinycuda_blas ${OPENBLAS_LIBRARIES})
