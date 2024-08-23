#include "cuda_kernel_generator.h"
#include <fstream>
#include <sstream>

std::string CUDAKernelGenerator::generate(const std::vector<ONNXNode>& graph) {
    std::stringstream cuda_code;
    
    for (const auto& node : graph) {
        if (node.op_type == "Conv") {
            cuda_code << generateConv2D(node);
        } else if (node.op_type == "Relu") {
            cuda_code << generateReLU(node);
        } else if (node.op_type == "MaxPool") {
            cuda_code << generateMaxPool(node);
        }
    }
    
    return cuda_code.str();
}

std::string CUDAKernelGenerator::generateConv2D(const ONNXNode& node) {
    std::ifstream t("cuda/kernel_templates/conv2d.cu");
    std::string cuda_template((std::istreambuf_iterator<char>(t)),
                               std::istreambuf_iterator<char>());
    return cuda_template;
}

std::string CUDAKernelGenerator::generateReLU(const ONNXNode& node) {
    return "// ReLU kernel\n";
}

std::string CUDAKernelGenerator::generateMaxPool(const ONNXNode& node) {
    return "// MaxPool kernel\n";
}
