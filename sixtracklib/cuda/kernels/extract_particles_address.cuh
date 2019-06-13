#ifndef SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDRESS_KERNEL_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDRESS_KERNEL_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(Particles_extract_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC void* SIXTRL_RESTRICT raw_paddr_arg,
    SIXTRL_BUFFER_DATAPTR_DEC void* SIXTRL_RESTRICT particles_buffer_arg,
    uint64_t const slot_size );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */


#endif /* SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDRESS_KERNEL_CUH__ */
