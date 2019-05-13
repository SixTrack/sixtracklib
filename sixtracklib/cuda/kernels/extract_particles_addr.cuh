#ifndef SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDR_KERNELS_CUH__
#define SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDR_KERNELS_CUH__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__global__ void NS(Particles_buffer_store_all_addresses_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    NS(buffer_size_t) const slot_size );

__global__ void NS(Particles_buffer_store_all_addresses_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SXITRL_RESTRICT ptr_dbg_register);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_KERNELS_EXTRACT_PARTICLES_ADDR_KERNELS_CUH__ */

/* end sixtracklib/cuda/kernels/extract_particles_addr_kernels.cuh */
