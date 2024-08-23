#pragma once
#include <string>
#include <vector>
#include "onnx_parser.h"

class CUDAKernelGenerator {
public:
    std::string generate(const std::vector<ONNXNode>& graph);
private:
    std::string generateConv2D(const ONNXNode& node);
    std::string generateReLU(const ONNXNode& node);
    std::string generateMaxPool(const ONNXNode& node);
};
