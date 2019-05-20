#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/extract_particles_addr.cuh"
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
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


__global__ void NS(Particles_buffer_store_all_addresses_cuda)(
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

__global__ void NS(Particles_buffer_store_all_addresses_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(arch_debugging_t) debug_register_t;

    debug_register_t dbg = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;

    buf_size_t const num_objects = NS(ManagedBuffer_get_num_objects)(
        pbuffer, slot_size );

    buf_size_t thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    buf_size_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    for( ; thread_id < num_objects ; thread_id += stride )
    {
        NS(Particles_managed_buffer_store_addresses_debug)(
            pbuffer, paddr_buffer, thread_id, slot_size, &dbg );

        if( dbg != SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY )
        {
            break;
        }
    }

    if( ( ptr_dbg_register != SIXTRL_NULLPTR ) &&
        ( dbg != SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY ) )
    {
        *ptr_dbg_register |= dbg;
    }
}

/* end: /cuda/kernels/extract_particles_address.cu */
