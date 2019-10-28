#ifndef SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"
    #include "sixtracklib/opencl/internal/status_flag.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ManagedBuffer_remap_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        NS(arch_status_t) const status =
            NS(ManagedBuffer_remap)( buffer, slot_size );

        if( ptr_status_flag != SIXTRL_NULLPTR )
        {
            NS(arch_debugging_t) flags = ( NS(arch_debugging_t) )0u;

            if( status != SIXTRL_ARCH_STATUS_SUCCESS )
            {
                if( buffer == SIXTRL_NULLPTR )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( slot_size == ( buf_size_t )0u )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( NS(ManagedBuffer_needs_remapping)( buffer, slot_size ) )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );
            }

            *ptr_status_flag = NS(DebugReg_store_arch_status)( flags, status );
        }
    }
}

__kernel void NS(ManagedBuffer_remap_io_buffers_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_UINT64_T const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flag )
{
    size_t const global_id = ( size_t )get_global_id( 0 );

    size_t const gid_to_remap_input_buffer  = ( size_t )0u;
    size_t const gid_to_remap_output_buffer =
        ( get_global_size( 0 ) > 1u ) ? ( size_t )1u : ( size_t )0u;

    if( global_id <= gid_to_remap_output_buffer )
    {
        NS(arch_status_t) status = SIXTRL_ARCH_STATUS_SUCCESS;
        NS(arch_debugging_t) flags = ( NS(arch_debugging_t) )0u;

        if( global_id == gid_to_remap_input_buffer )
        {
            status =  NS(ManagedBuffer_remap)( in_buffer_begin, slot_size );

            if( status != SIXTRL_ARCH_STATUS_SUCCESS )
            {
                if( in_buffer_begin == SIXTRL_NULLPTR )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( slot_size == ( size_t )0u )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( NS(ManagedBuffer_needs_remapping)(
                        in_buffer_begin, slot_size ) )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                flags = NS(DebugReg_store_arch_status)( flags, status );
            }
        }

        if( global_id == gid_to_remap_output_buffer )
        {
            status = NS(ManagedBuffer_remap)( out_buffer_begin, slot_size );

            if( status != SIXTRL_ARCH_STATUS_SUCCESS )
            {
                if( out_buffer_begin == SIXTRL_NULLPTR )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( slot_size == ( size_t )0u )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                if( NS(ManagedBuffer_needs_remapping)(
                        out_buffer_begin, slot_size ) )
                    flags = NS(DebugReg_raise_next_error_flag)( flags );

                flags = NS(DebugReg_store_arch_status)( flags, status );
            }
        }

        if( ptr_status_flag != SIXTRL_NULLPTR )
        {
            NS(OpenCl1x_collect_status_flag_value)( ptr_status_flag, flags );
        }
    }
}


__kernel void NS(ManagedBuffer_needs_remapping_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_UINT64_T const slot_size,
    NS(arch_debugging_t) const needs_remapping_true,
    NS(arch_debugging_t) const needs_remapping_false,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_result )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_ASSERT( buffer_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( buf_size_t )0u );

    if( ( buf_size_t )get_global_id( 0 ) == ( buf_size_t )0u )
    {
        bool const needs_remapping =
            NS(ManagedBuffer_needs_remapping)( buffer_begin, slot_size );

        if( ptr_result != SIXTRL_NULLPTR )
        {
            *ptr_result = ( needs_remapping )
                ? needs_remapping_true : needs_remapping_false;
        }
    }
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */

/* end: sixtracklib/opencl/kernels/managed_buffer_remap_debug.cl */
