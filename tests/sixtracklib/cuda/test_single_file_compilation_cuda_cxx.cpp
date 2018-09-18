#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "tests/sixtracklib/cuda/impl/cuda_grid_dimensions_kernel.cuh"

TEST( CXX_Cuda_SingleFileCompilation,
      IterateOverAllDevicesTestGridDimensions )
{
    dim3 const grid_dimensions( 8, 8 );
    dim3 const block_dimensions( 4 );

    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount( &num_devices );
    ASSERT_TRUE( err == cudaSuccess );

    for( int device = 0 ; device < num_devices ; ++device )
    {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties( &deviceProp, device);
        ASSERT_TRUE( err == cudaSuccess );

        std::cout << deviceProp.major << "."
                  << deviceProp.minor << std::endl;

        err = cudaSetDevice( device );
        ASSERT_TRUE( err == cudaSuccess );

        runTestDimensions( grid_dimensions, block_dimensions );
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        ASSERT_TRUE( err == cudaSuccess );

        cudaStream_t stream;
        err = cudaStreamCreate( &stream );
        ASSERT_TRUE( err == cudaSuccess );


        const void* ptr_kernel = reinterpret_cast< void* >( &TestDimensions );

        err = cudaLaunchKernel( ptr_kernel, grid_dimensions, block_dimensions,
                nullptr, ::size_t{ 0 }, stream );

        ASSERT_TRUE( err == cudaSuccess );

        err = cudaStreamSynchronize( stream );
        ASSERT_TRUE( err == cudaSuccess );

        err = cudaStreamDestroy( stream );
        ASSERT_TRUE( err == cudaSuccess );
    }
}

/* end: tests/sixtracklib/cuda/test_single_file_compilation_cuda_cxx.cpp */
