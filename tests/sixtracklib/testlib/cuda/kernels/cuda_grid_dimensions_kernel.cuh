#ifndef SIXTRACKLIB_TESTLIB_CUDA_KERNELS_GRID_DIMENSIONS_KERNEL_HEADER_CUH__
#define SIXTRACKLIB_TESTLIB_CUDA_KERNELS_GRID_DIMENSIONS_KERNEL_HEADER_CUH__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>

    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

__global__ void TestDimensions();

#if !defined( SIXTRL_CUDA_NVRTC )
__host__ void runTestDimensions( dim3 gridDim, dim3 blockDim );
#endif /* !defined( SIXTRL_CUDA_NVRTC ) */

#endif /* SIXTRACKLIB_TESTLIB_CUDA_KERNELS_GRID_DIMENSIONS_KERNEL_HEADER_CUH__ */

/* end: tests/sixtracklib/testlib/cuda/kernels/cuda_grid_dimensions_kernel.cuh */
