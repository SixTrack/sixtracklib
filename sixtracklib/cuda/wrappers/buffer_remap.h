#ifndef SIXTRACKLIB_CUDA_WRAPPERS_BUFFER_REMAP_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_BUFFER_REMAP_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Buffer_remap_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Buffer_remap_debug_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ManagedBuffer_remap_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ManagedBuffer_remap_debug_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_BUFFER_REMAP_H__ */
/* end: sixtracklib/cuda/wrappers/buffer_remap.h */
