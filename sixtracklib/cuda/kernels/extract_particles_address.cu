#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/extract_particles_address.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>

    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Particles_extract_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC void* SIXTRL_RESTRICT paddr_arg,
    SIXTRL_BUFFER_DATAPTR_DEC void* SIXTRL_RESTRICT pbuffer_arg,
    uint64_t const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_paddr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char*            ptr_raw_t;

    ptr_paddr_t paddr = ( ptr_paddr_t )paddr_arg;
    ptr_raw_t particles_buffer = ( ptr_raw_t )pbuffer_arg;

    SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* particles =
        NS(Particles_managed_buffer_get_const_particles)( particles_buffer,
            0, slot_size );

    NS(Particles_store_addresses)( paddr, particles );
    return;
}

/* end: */
