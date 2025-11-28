#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

template <typename T>
void cudaMemcpySafe(T* dst, const T* src, size_t size, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
}

template <typename T>
T* allocateDeviceMemory(size_t size) {
    T* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    return d_ptr;
}

template <typename T>
void freeDeviceMemory(T* d_ptr) {
    CUDA_CHECK(cudaFree(d_ptr));
}

#endif