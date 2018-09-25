#ifndef SIXTRL_TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_HEADER_CUH__
#define SIXTRL_TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_HEADER_CUH__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(BeamElements_copy_beam_elements_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_ARGPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRL_TESTS_SIXTRACKLIB_CUDA_IMPL_CUDA_BEAM_ELEMENTS_KERNEL_HEADER_CUH__ */

/* end: tests/sixtracklib/cuda/impl/cuda_beam_elements_kernel.cuh */
