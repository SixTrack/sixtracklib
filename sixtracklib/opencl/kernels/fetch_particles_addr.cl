#ifndef SIXTRACKLIB_OPENCL_KERNELS_FETCH_PARTICLES_ADDR_CL__
#define SIXTRACKLIB_OPENCL_KERNELS_FETCH_PARTICLES_ADDR_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/particles/particles_addr.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Particles_buffer_store_all_addresses_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT paddr_buffer_begin,
    SIXTRL_UINT64_T const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_objects = NS(ManagedBuffer_get_num_objects)(
        pbuffer_begin, slot_size );

    buf_size_t ii = ( buf_size_t )get_global_id( 0 );
    buf_size_t const stride = ( buf_size_t )get_global_size( 0 );

    for( ; ii < num_objects ; ii += stride )
    {
        NS(Particles_managed_buffer_store_addresses)(
            paddr_buffer_begin, pbuffer_begin, ii, slot_size );
    }
}

#endif /* SIXTRACKLIB_OPENCL_KERNELS_FETCH_PARTICLES_ADDR_CL__ */

