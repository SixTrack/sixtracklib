#ifndef SIXTRACKLIB_CUDA_WRAPPERS_EXTRACT_PARTICLES_ADDRESSES_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_EXTRACT_PARTICLES_ADDRESSES_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(controller_status_t)
NS(Particles_extract_addresses_cuda_on_grid)(
    void* SIXTRL_RESTRICT addr_arg_buffer,
    void* SIXTRL_RESTRICT pbuffer_arg_buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const num_threads_per_block );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(controller_status_t)
NS(Particles_extract_addresses_cuda)(
    void* SIXTRL_RESTRICT addr_arg_buffer,
    void* SIXTRL_RESTRICT pbuffer_arg_buffer );

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_EXTRACT_PARTICLES_ADDRESSES_H__ */

/* end: sixtracklib/cuda/wrappers/extract_particles_address.h */
