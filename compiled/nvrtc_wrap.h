#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <vector>
#include <iostream>

#include "utils.h"

// GPU occupancy tuning
//const size_t NUM_THREADS = 256;
const size_t NUM_THREADS = 1024;
//const size_t NUM_BLOCKS = 512; 
const size_t NUM_BLOCKS = 1024; 

// Headers for the cuda code
std::vector<std::string> my_headers_names {
"track.h"
};

std::vector<std::string> my_headers {{
#include "track.xxd"
}};


static void CUDA_info() {
  using namespace std;
  const int kb = 1024;
  const int mb = kb * kb;
  cout << "NBody.GPU" << endl << "=========" << endl << endl;
  
  cout << "CUDA version:   v" << CUDART_VERSION << endl;

  int devCount;
  cudaGetDeviceCount(&devCount);
  cout << "CUDA Devices: " << endl << endl;
  
  for(int i = 0; i < devCount; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
    cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
    cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
    cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
    cout << "  Block registers: " << props.regsPerBlock << endl << endl; 

    cout << "  Warp size:         " << props.warpSize << endl;
    cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
    cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
    cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
    cout << endl;
  }
}

std::string CUDA_get_architecture(int device) {
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  return std::to_string(props.major) + std::to_string(props.minor);
}

template <typename T>
static void CUDA_SAFE_CALL(const T & x) { 
  CUresult result = x;
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::cerr << "\nerror: " << x << " failed with error "
              << msg << '\n';
    exit(1);
  }
}

class CUDA {
  public:
  CUdevice cuDevice;
  CUcontext context;

  CUDA() {
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  }

  ~CUDA() {
    CUDA_SAFE_CALL(cuCtxDestroy(context));
  }
} cuda;

class NVRTC {

  CUmodule module;
  CUfunction kernel;

  std::vector<std::string> compilation_opts;
  std::vector<char> ptx;
  public:

  template <typename T>
  static void SAFE_CALL(const T & x) {
    nvrtcResult result = x;
    if (result != NVRTC_SUCCESS) {
      std::cerr << "\nerror: " << x << " failed with error "
                << nvrtcGetErrorString(result) << '\n';
      exit(1);
    }
  }

  NVRTC() {
    compilation_opts.emplace_back("--gpu-architecture=compute_"+CUDA_get_architecture(0));
//    compilation_opts.emplace_back("-I .");
//    compilation_opts.emplace_back("-I ./include");
    compilation_opts.emplace_back("--fmad=false");
    compilation_opts.emplace_back("--std=c++11");
  }

  NVRTC(const std::string & ptx_path): NVRTC() {
    read_ptx(ptx_path);
std::cout << "loaded ptx " << ptx_path << std::endl;
  }
  
  NVRTC(const NVRTC & o) = delete;
  NVRTC & operator=(const NVRTC & o) = delete;
  NVRTC(NVRTC && o) = default;
  NVRTC & operator=(NVRTC && o) = default;

  ~NVRTC() {
    clear();
  }

  void compile_ptx(std::string buffer) {
    //cuCtxSetCurrent(cuda.context); 
    auto ptr_headers = vecStr2vec<const char*>(my_headers);
    auto ptr_headers_names = vecStr2vec<const char*>(my_headers_names);

    nvrtcProgram prog;
    NVRTC::SAFE_CALL(
      nvrtcCreateProgram(&prog,          // prog
                         buffer.c_str(), // buffer
                         "lattice.cu",   // name
                         ptr_headers.size(),          // numHeaders
                         ptr_headers.data(),         // headers
                         ptr_headers_names.data())); // includeNames
    // Compile the program
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    compilation_opts.size(),     // numOptions
                                                    vecStr2vec<const char *>(compilation_opts).data()); // options
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC::SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
      char *log = new char[logSize];
      NVRTC::SAFE_CALL(nvrtcGetProgramLog(prog, log));
      std::cout << log << '\n';
      delete[] log;
    }
    if (compileResult != NVRTC_SUCCESS) {
      exit(1);
    }
    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC::SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx.resize(ptxSize);
    NVRTC::SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
    // Destroy the program.
    NVRTC::SAFE_CALL(nvrtcDestroyProgram(&prog));
    load_ptx();
  }

  void load_ptx() {
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "track"));
  }

  void write_ptx(const std::string & path) {
    std::ofstream file(path, std::ios::out | std::ofstream::binary | std::ofstream::trunc);
    std::copy(ptx.begin(), ptx.end(), std::ostreambuf_iterator<char>(file));
  }

  void read_ptx(const std::string & path) {
    std::ifstream file(path, std::ios::in | std::ifstream::binary);
    if ( file.is_open() ) {
      clear();
      std::istreambuf_iterator<char> iter(file);
      std::copy(iter, std::istreambuf_iterator<char>(), std::back_inserter(ptx));
      load_ptx();
    } else {
      throw std::runtime_error("cannot find "+path);
    }
  }

  void run(void** args) {
    if (ptx.empty()) throw std::runtime_error("Compile or load the ptx before trying to run it!");

    //cuCtxSetCurrent(cuda.context); 

    CUDA_SAFE_CALL(
      cuLaunchKernel(kernel,
                     NUM_THREADS, 1, 1,   // grid dim
                     NUM_BLOCKS, 1, 1,    // block dim
                     0, NULL,             // shared mem and stream
                     args, 0));    // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
  }

  void clear() {
    if (!ptx.empty()) {
      CUDA_SAFE_CALL(cuModuleUnload(module));
      ptx.clear();
    }
  }
};

