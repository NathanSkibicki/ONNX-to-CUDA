#pragma once
#include <cuda_runtime.h>

class PerformanceMetrics {
public:
    void startTimer();
    float endTimer();
    float getGPUUtilization();
    float getThroughput(int batch_size);
};