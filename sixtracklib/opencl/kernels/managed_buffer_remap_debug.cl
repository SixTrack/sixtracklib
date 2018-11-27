#ifndef SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(ManagedBuffer_remap_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(ManagedBuffer_remap_io_buffers_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(ManagedBuffer_remap_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id           = get_global_id( 0 );
    size_t const gid_to_remap_buffer = ( size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        long int success_flag = ( buffer_begin != SIXTRL_NULLPTR ) ? 0 : -1;

        buf_size_t const slot_size = ( buf_size_t )8u;

        if( 0 != NS(ManagedBuffer_remap)( buffer_begin, slot_size ) )
        {
            success_flag |= -2;
        }

        if( ptr_success_flag != SIXTRL_NULLPTR )
        {
            atomic_or( ptr_success_flag, success_flag );
        }
    }

    return;
}

__kernel void NS(ManagedBuffer_remap_io_buffers_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id = get_global_id( 0 );
    size_t const gid_to_remap_input_buffer  = ( size_t )0u;
    size_t const gid_to_remap_output_buffer = ( get_global_size( 0 ) > 1u )
        ? ( gid_to_remap_input_buffer + 1u )
        : ( gid_to_remap_input_buffer );

    if( global_id <= gid_to_remap_output_buffer )
    {
        long int success_flag = ( long int )0;
        buf_size_t const slot_size = ( buf_size_t )8u;

        if( global_id == gid_to_remap_input_buffer )
        {
            if( ( in_buffer_begin != SIXTRL_NULLPTR ) &&
                ( in_buffer_begin != out_buffer_begin ) )
            {
                if( 0 != NS(ManagedBuffer_remap)( in_buffer_begin, slot_size ) )
                {
                    success_flag |= -2;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( global_id == gid_to_remap_output_buffer )
        {
            if( ( out_buffer_begin != SIXTRL_NULLPTR ) &&
                ( out_buffer_begin != in_buffer_begin ) )
            {
                if( 0 != NS(ManagedBuffer_remap)(
                        out_buffer_begin, slot_size ) )
                {
                    success_flag |= -4;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( ptr_success_flag != SIXTRL_NULLPTR )
        {
            atomic_or( ptr_success_flag, success_flag );
        }
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */

/* end: sixtracklib/opencl/kernels/managed_buffer_remap_debug.cl */
