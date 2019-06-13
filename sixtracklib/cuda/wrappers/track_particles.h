#ifndef SIXTRACKLIB_CUDA_WRAPPERS_TRACK_PARTICLES_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_TRACK_PARTICLES_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t)
NS(Track_particles_line_cuda_on_grid)(
    void* SIXTRL_RESTRICT particles_buffer_arg,
    void* SIXTRL_RESTRICT beam_elements_buffer_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Track_particles_line_cuda)(
    void* SIXTRL_RESTRICT particles_buffer_arg,
    void* SIXTRL_RESTRICT beam_elements_buffer_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_EXTRACT_PARTICLES_ADDRESSES_H__ */

/* end: sixtracklib/cuda/wrappers/extract_particles_address.h */
