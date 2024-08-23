#pragma once
#include <string>
#include <cuda.h>

class JITCompiler {
public:
    CUmodule compile(const std::string& cuda_code);
private:
};
