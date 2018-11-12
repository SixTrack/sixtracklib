#ifndef SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ManagedBuffer_remap_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin );

__kernel void NS(ManagedBuffer_remap_io_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin );

/* ========================================================================= */

__kernel void NS(ManagedBuffer_remap_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id           = get_global_id( 0 );
    size_t const gid_to_remap_buffer = ( size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        NS(ManagedBuffer_remap)( buffer_begin, ( buf_size_t )8u );
    }

    return;
}

__kernel void NS(ManagedBuffer_remap_io_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin )
{
    size_t const global_id = get_global_id( 0 );
    size_t const gid_to_remap_input_buffer  = ( size_t )0u;

    size_t const gid_to_remap_output_buffer = ( get_global_size( 0 ) > 1u )
        ? ( gid_to_remap_input_buffer + 1u )
        : ( gid_to_remap_input_buffer );

    if( global_id <= gid_to_remap_output_buffer )
    {
        SIXTRL_ASSERT( in_buffer_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( in_buffer_begin != out_buffer_begin );

        if( global_id == gid_to_remap_input_buffer )
        {
            NS(ManagedBuffer_remap)( in_buffer_begin, ( NS(buffer_size_t) )8u );
        }

        if( global_id == gid_to_remap_output_buffer )
        {
            NS(ManagedBuffer_remap)( out_buffer_begin, ( NS(buffer_size_t) )8u );
        }
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */

/* end: sixtracklib/opencl/kernels/managed_buffer_remap.cl */
