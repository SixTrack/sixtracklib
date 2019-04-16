#ifndef SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__
#define SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(Track_particles_line)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    uint64_t const line_begin_idx,
    uint64_t const line_end_idx,
    bool const finish_turn );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_IMPL_TRACK_PARTICLES_KERNEL_CUDA_HEADER_CUH__ */

/* end sixtracklib/cuda/impl/track_particles_kernel.cuh */
