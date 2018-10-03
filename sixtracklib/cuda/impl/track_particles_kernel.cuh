#ifndef SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__
#define SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(Remap_particles_beam_elements_buffers_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

__global__ void NS(Track_particles_beam_elements_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elem_buf,
    SIXTRL_UINT64_T const num_turns,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );


#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__ */

/* end sixtracklib/cuda/impl/track_particles_kernel.cuh */
