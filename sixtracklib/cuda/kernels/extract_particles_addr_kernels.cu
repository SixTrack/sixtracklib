#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/extract_particles_addr_kernels.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


__global__ void NS(Cuda_fetch_particle_addresses_kernel)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_objects = NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size );

    buf_size_t thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    buf_size_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    for( ; thread_id < num_objects ; thread_id += stride )
    {
        NS(Particles_managed_buffer_store_addresses)(
            particles_buffer, paddr_buffer, thread_id, slot_size );
    }
}

__global__ void NS(Cuda_fetch_particle_addresses_debug_kernel)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(ctrl_debug_flag_t)* SIXTRL_RESTRICT debug_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(ctrl_debug_flag_t) debug_flag_t;

    debug_flag_t debug_flag = SIXTRL_CONTROLLER_DEBUG_FLAG_OK;

    buf_size_t const num_objects = NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size );

    buf_size_t thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    buf_size_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    for( ; thread_id < num_objects ; thread_id += stride )
    {
        NS(Particles_managed_buffer_store_addresses_debug)(
            pbuffer, paddr_buffer, thread_id, slot_size, &debug_flag );

        if( debug_flag != SIXTRL_CONTROLLER_DEBUG_FLAG_OK )
        {
            break;
        }
    }

    NS(Cuda_handle_debug_flag_in_kernel)( ptr_debug_flag, debug_flag );
}

/* end: /cuda/kernels/extract_particles_address.cu */
