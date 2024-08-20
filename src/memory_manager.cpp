#include "memory_manager.h"
#include <stdexcept>

void* MemoryManager::allocateGPU(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    return ptr;
}

void MemoryManager::freeGPU(void* ptr) {
    cudaFree(ptr);
}

void MemoryManager::copyToGPU(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to GPU");
    }
}

void MemoryManager::copyFromGPU(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from GPU");
    }
}