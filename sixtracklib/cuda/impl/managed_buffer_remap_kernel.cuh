#ifndef SIXTRACKLIB_CUDA_IMPL_MANAGED_BUFFER_REMAP_KERNEL_CUDA_HEADER_CUH__
#define SIXTRACKLIB_CUDA_IMPL_MANAGED_BUFFER_REMAP_KERNEL_CUDA_HEADER_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(ManagedBuffer_remap_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

__global__ void NS(ManagedBuffer_remap_io_buffers_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_MANAGED_BUFFER_REMAP_KERNEL_CUDA_HEADER_CUH__ */

/* end: sixtracklib/cuda/impl/managed_buffer_remap_kernel.cuh */
