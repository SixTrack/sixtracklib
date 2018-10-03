#ifndef SIXTRACKLIB_TESTLIB_CUDA_KERNELS_BUFFER_GENERIC_OBJ_KERNEL_HEADER_CUH__
#define SIXTRACKLIB_TESTLIB_CUDA_KERNELS_BUFFER_GENERIC_OBJ_KERNEL_HEADER_CUH__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Remap_original_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT orig_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

__global__ void NS(Copy_original_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*       SIXTRL_RESTRICT copy_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

#endif /* SIXTRACKLIB_TESTLIB_CUDA_KERNELS_BUFFER_GENERIC_OBJ_KERNEL_HEADER_CUH__ */

/* end: tests/sixtracklib/testlib/cuda/kernels/cuda_buffer_generic_obj_kernel.cuh */
