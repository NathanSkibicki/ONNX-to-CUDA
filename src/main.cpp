#include <iostream>
#include <vector>
#include "onnx_parser.h"
#include "cuda_kernel_generator.h"
#include "jit_compiler.h"
#include "memory_manager.h"
#include "performance_metrics.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    
    ONNXParser parser;
    auto graph = parser.parse(model_path);
    
    CUDAKernelGenerator generator;
    auto cuda_code = generator.generate(graph);
    
    JITCompiler compiler;
    auto compiled_module = compiler.compile(cuda_code);
    
    // Prepare input data
    std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);  // Example input size
    
    MemoryManager mem_manager;
    void* d_input = mem_manager.allocateGPU(input_data.size() * sizeof(float));
    void* d_output = mem_manager.allocateGPU(1000 * sizeof(float));  // Assuming 1000 output classes
    
    mem_manager.copyToGPU(d_input, input_data.data(), input_data.size() * sizeof(float));
    
    // Run the compiled module
    PerformanceMetrics metrics;
    metrics.startTimer();
    
    compiled_module.run(d_input, d_output);
    
    float elapsed_time = metrics.endTimer();
    
    // Copy results back to host
    std::vector<float> output_data(1000);
    mem_manager.copyFromGPU(output_data.data(), d_output, output_data.size() * sizeof(float));
    
    // Print results
    std::cout << "Inference completed in " << elapsed_time << " ms" << std::endl;
    std::cout << "GPU Utilization: " << metrics.getGPUUtilization() << "%" << std::endl;
    std::cout << "Throughput: " << metrics.getThroughput(1) << " images/second" << std::endl;
    
    // Clean up
    mem_manager.freeGPU(d_input);
    mem_manager.freeGPU(d_output);
    
    return 0;
}