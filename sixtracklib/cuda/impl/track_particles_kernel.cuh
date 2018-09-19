#ifndef SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__
#define SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

extern __global__ void NS(Remap_particles_beam_elements_buffers_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

extern __global__ void NS(Track_particles_beam_elements_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buf,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buf,
    SIXTRL_UINT64_T const num_turns,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

#endif /* SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__ */

/* end sixtracklib/cuda/impl/track_particles_kernel.cuh */
