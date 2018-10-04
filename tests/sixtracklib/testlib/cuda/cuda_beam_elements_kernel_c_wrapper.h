#ifndef TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_C_WRAPPER_H__
#define TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_C_WRAPPER_H__

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

SIXTRL_HOST_FN int NS(Run_test_copy_beam_elements_buffer_kernel_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const threads_per_block );

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

SIXTRL_HOST_FN int NS(Run_test_copy_beam_elements_buffer_kernel_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer );

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_C_WRAPPER_H__ */

/* end: tests/sixtracklib/cuda/impl/cuda_particles_kernel_c_wrapper.h */
