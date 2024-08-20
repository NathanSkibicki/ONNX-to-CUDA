#include "onnx_parser.h"
#include <fstream>
#include <onnx/onnx_pb.h>

std::vector<ONNXNode> ONNXParser::parse(const std::string& model_path) {
    ONNX_NAMESPACE::ModelProto model;
    std::ifstream ifs(model_path, std::ios::binary);
    if (!model.ParseFromIstream(&ifs)) {
        throw std::runtime_error("Failed to parse ONNX model");
    }
    
    std::vector<ONNXNode> graph;
    for (const auto& node : model.graph().node()) {
        ONNXNode onnx_node;
        onnx_node.op_type = node.op_type();
        onnx_node.inputs = std::vector<std::string>(node.input().begin(), node.input().end());
        onnx_node.outputs = std::vector<std::string>(node.output().begin(), node.output().end());
        graph.push_back(onnx_node);
    }
    
    return graph;
}