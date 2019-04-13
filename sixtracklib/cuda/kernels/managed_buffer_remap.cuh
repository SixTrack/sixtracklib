#ifndef SIXTRACKLIB_CUDA_KERNEL_MANAGED_BUFFER_REMAP_KERNEL_CUH__
#define SIXTRACKLIB_CUDA_KERNEL_MANAGED_BUFFER_REMAP_KERNEL_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(ManagedBuffer_remap_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    uint64_t const slot_size );

__global__ void NS(ManagedBuffer_remap_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    uint64_t const slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_MANAGED_BUFFER_REMAP_KERNEL_CUDA_HEADER_CUH__ */

/* end: sixtracklib/cuda/impl/managed_buffer_remap_kernel.cuh */
