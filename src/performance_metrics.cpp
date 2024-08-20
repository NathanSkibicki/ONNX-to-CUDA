#include "performance_metrics.h"
#include <cuda_runtime.h>
#include <nvml.h>

void PerformanceMetrics::startTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float PerformanceMetrics::endTimer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float PerformanceMetrics::getGPUUtilization() {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(device, &utilization);
    nvmlShutdown();
    return utilization.gpu;
}

float PerformanceMetrics::getThroughput(int batch_size) {
    float elapsed_time = endTimer();
    return (batch_size * 1000.0f) / elapsed_time;  // images per second
}