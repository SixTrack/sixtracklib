#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/_impl/path.h"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

TEST( CXX_Cuda_RunTimeCompilation,
      IterateOverAllDevicesTestGridDimensions )
{
    nvrtcProgram test_program;

    std::string const PATH_BASE_DIR = ::st_PATH_TO_BASE_DIR;

    std::string const path_kernel_file( PATH_BASE_DIR + std::string(
        "tests/sixtracklib/cuda/details/cuda_grid_dimensions_kernel.cu" ) );

    std::ifstream source_file = std::ifstream( path_kernel_file, std::ios::in );

    std::string const KERNEL_PROGRAM_SOURCE(
        ( std::istreambuf_iterator< char >( source_file ) ),
          std::istreambuf_iterator< char >() );

    nvrtcResult err = nvrtcCreateProgram(
        &test_program, KERNEL_PROGRAM_SOURCE.c_str(),
        "test_runtime_compilation_program",
        0, nullptr, nullptr );

    ASSERT_TRUE( err == NVRTC_SUCCESS );


    char const KERNEL_FN_NAME[] = "&TestDimensions";
    err = nvrtcAddNameExpression( test_program, KERNEL_FN_NAME );
    ASSERT_TRUE( err == NVRTC_SUCCESS );

    std::vector< std::string > compile_options;

    compile_options.push_back( std::string( "-D_GPUCODE=1" ) );
    compile_options.push_back( std::string( "-D_FORCE_INLINES" ) );
    compile_options.push_back( std::string( "-DSIXTRL_NO_SYSTEM_INCLUDES=1" ) );
    compile_options.push_back( std::string( "-DSIXTRL_NO_INCLUDES=1" ) );
    compile_options.push_back( std::string( "-DSIXTRL_CUDA_NVRTC=1" ) );
    compile_options.push_back( std::string( "--device-c" ) );
    compile_options.push_back( std::string( "--gpu-architecture=compute_30" ) );
//     compile_options.push_back( std::string( "--gpu-architecture=compute_60" ) );
//     compile_options.push_back( std::string( "--gpu-architecture=compute_61" ) );
    compile_options.push_back( std::string( "-G" ) );
    compile_options.push_back( std::string( "--generate-line-info" ) );

    std::vector< char* > compile_options_cstr( compile_options.size(), nullptr );
    compile_options_cstr.clear();

    for( auto& opt : compile_options )
    {
        compile_options_cstr.push_back( const_cast< char* >( opt.c_str() ) );
    }

    err = nvrtcCompileProgram( test_program,
       compile_options_cstr.size(), compile_options_cstr.data() );

    size_t log_size = size_t{ 0 };
    nvrtcResult err2 = nvrtcGetProgramLogSize( test_program, &log_size );

    ASSERT_TRUE( err2 == NVRTC_SUCCESS );

    if( log_size > size_t{ 0 } )
    {
        std::vector< char > log_str( log_size + 1u, '\0' );
        err2 = nvrtcGetProgramLog( test_program, log_str.data() );

        std::cout << "Compile log: \r\n"
                  << log_str.data() << std::endl;
    }

    ASSERT_TRUE( err == NVRTC_SUCCESS );

    const char* mangled_fn_name = nullptr;
    err = nvrtcGetLoweredName( test_program, KERNEL_FN_NAME, &mangled_fn_name );

    ASSERT_TRUE( mangled_fn_name != nullptr );
    ASSERT_TRUE( err == NVRTC_SUCCESS );

    size_t ptx_size = size_t{ 0 };
    err = nvrtcGetPTXSize( test_program, &ptx_size );

    ASSERT_TRUE( err == NVRTC_SUCCESS );
    ASSERT_TRUE( ptx_size > size_t{ 0 } );

    std::vector< char > ptx_bin_store( ptx_size, int8_t{ 0 } );
    err = nvrtcGetPTX( test_program, ptx_bin_store.data() );
    ASSERT_TRUE( err == NVRTC_SUCCESS );

    CUresult cu_err = cuInit( 0 );
    ASSERT_TRUE( cu_err == CUDA_SUCCESS );

    int num_devices = 0;
    cu_err = cuDeviceGetCount ( &num_devices );
    ASSERT_TRUE( cu_err == CUDA_SUCCESS );

    for( int device_id = 0 ; device_id < num_devices ; ++device_id )
    {
        CUdevice    device;
        CUcontext   context;
        CUmodule    module;
        CUfunction  kernel;

        cu_err = cuDeviceGet( &device, device_id );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuCtxCreate( &context, 0, device );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuModuleLoadDataEx(
            &module, ptx_bin_store.data(), 0, nullptr, nullptr );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuModuleGetFunction( &kernel, module, mangled_fn_name );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        dim3 const grid_dim( 8, 8 );
        dim3 const block_dim( 4 );

        cu_err = cuLaunchKernel( kernel,
                                 grid_dim.x, grid_dim.y, grid_dim.z,
                                 block_dim.x, block_dim.y, block_dim.z,
                                 0, nullptr, nullptr, nullptr );

        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuCtxSynchronize();
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuModuleUnload( module );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );

        cu_err = cuCtxDestroy( context );
        ASSERT_TRUE( cu_err == CUDA_SUCCESS );
    }

    err = nvrtcDestroyProgram( &test_program );
    ASSERT_TRUE( err == NVRTC_SUCCESS );
}

/* end: tests/sixtracklib/cuda/test_runtime_compilation_cuda_cxx.cpp */
