#pragma once
#include <cuda_runtime.h>

class MemoryManager {
public:
    void* allocateGPU(size_t size);
    void freeGPU(void* ptr);
    void copyToGPU(void* dst, const void* src, size_t size);
    void copyFromGPU(void* dst, const void* src, size_t size);
};