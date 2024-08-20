#include "jit_compiler.h"
#include <nvrtc.h>
#include <cuda.h>
#include <stdexcept>

CUmodule JITCompiler::compile(const std::string& cuda_code) {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, cuda_code.c_str(), "generated_kernel.cu", 0, NULL, NULL);
    
    const char* opts[] = {"--gpu-architecture=compute_70", "--fmad=false"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);
    
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char* log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    std::cout << log << std::endl;
    delete[] log;
    
    if (compileResult != NVRTC_SUCCESS) {
        throw std::runtime_error("CUDA kernel compilation failed");
    }
    
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);
    
    CUmodule module;
    cuModuleLoadData(&module, ptx);
    
    delete[] ptx;
    nvrtcDestroyProgram(&prog);
    
    return module;
}