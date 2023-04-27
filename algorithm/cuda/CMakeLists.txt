cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(floyd CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(
        cuda_floyd floyd.cu
        # example_scalar_mult.cu
)


set_target_properties(cuda_floyd PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
