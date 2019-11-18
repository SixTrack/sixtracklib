#ifndef SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ManagedBuffer_remap_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_UINT64_T const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        SIXTRL_ASSERT( buffer_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( slot_size > ( buf_size_t )0u );

        NS(ManagedBuffer_remap)( buffer_begin, slot_size );
    }
}

__kernel void NS(ManagedBuffer_remap_io_buffers_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_UINT64_T const slot_size )
{
    size_t const global_id = ( size_t )get_global_id( 0 );

    size_t const gid_to_remap_input_buffer  = ( size_t )0u;
    size_t const gid_to_remap_output_buffer =
        ( get_global_size( 0 ) > 1u ) ? ( size_t )1u : ( size_t )0u;

    if( global_id <= gid_to_remap_output_buffer )
    {
        SIXTRL_ASSERT( in_buffer_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( in_buffer_begin != out_buffer_begin );
        SIXTRL_ASSERT( slot_size > ( buf_size_t )0u );

        if( global_id == gid_to_remap_input_buffer )
        {
            NS(ManagedBuffer_remap)( in_buffer_begin, slot_size );
        }

        if( global_id == gid_to_remap_output_buffer )
        {
            NS(ManagedBuffer_remap)( out_buffer_begin, slot_size );
        }
    }
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */

/* end: sixtracklib/opencl/kernels/managed_buffer_remap.cl */
