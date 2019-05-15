#ifndef SIXTRACKLIB_CUDA_KERNELS_MANAGED_BUFFER_REMAP_CUDA_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_MANAGED_BUFFER_REMAP_CUDA_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(ManagedBuffer_remap_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size );

__global__ void NS(ManagedBuffer_remap_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register );

__global__ void NS(ManagedBuffer_needs_remapping_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    NS(arch_debugging_t) const needs_remapping_true,
    NS(arch_debugging_t) const needs_remapping_false,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_result );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_KERNELS_MANAGED_BUFFER_REMAP_CUDA_CUH__ */

/* end: sixtracklib/cuda/kernels/managed_buffer_remap.cuh */
