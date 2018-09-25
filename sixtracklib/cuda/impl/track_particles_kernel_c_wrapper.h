#ifndef SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_C_WRAPPER_H__
#define SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_C_WRAPPER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN SIXTRL_STATIC int NS(Track_particles_in_place_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const num_turns,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

SIXTRL_HOST_FN int NS(Track_particles_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

SIXTRL_HOST_FN SIXTRL_STATIC int NS(Track_particles_in_place_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const num_turns );

SIXTRL_HOST_FN int NS(Track_particles_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns );

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Track_particles_in_place_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const num_turns,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block )
{
    return NS(Track_particles_on_cuda_grid)(
        particles, beam_elements, particles,
        num_turns, num_blocks, num_threads_per_block );
}

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

SIXTRL_INLINE int NS(Track_particles_in_place_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const num_turns )
{
    return NS(Track_particles_on_cuda)(
        particles, beam_elements, particles, num_turns );
}

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_C_WRAPPER_H__ */

/* end: /cuda/impl/track_particles_kernel_c_wrapper.h */
