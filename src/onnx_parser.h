#pragma once
#include <string>
#include <vector>
#include <onnx/onnx_pb.h>

struct ONNXNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // Add more fields as needed
};

class ONNXParser {
public:
    std::vector<ONNXNode> parse(const std::string& model_path);
private:
    // Helper methods
};