#ifndef SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__
#define SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(ManagedBuffer_remap_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_UINT64_T const buffer_slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC long int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const global_id           = get_global_id( 0 );
    size_t const gid_to_remap_buffer = ( size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        long int success_flag = ( buffer_begin != SIXTRL_NULLPTR ) ? 0 : -1;

        buf_size_t const slot_size =
            ( buffer_slot_size > ( SIXTRL_UINT64_T )0u )
                ? ( buf_size_t )buffer_slot_size : ( buf_size_t )8u;

        if( 0 != NS(ManagedBuffer_remap)( buffer_begin, slot_size ) )
        {
            success_flag |= -2;
        }

        if( ( success_flag == 0 ) &&
            ( NS(ManagedBuffer_needs_remapping)( buffer_begin, slot_size ) ) )
        {
            success_flag |= -4;
        }

        if(  ptr_success_flag != SIXTRL_NULLPTR )
        {
            *ptr_success_flag  = success_flag;
        }
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_IMPL_MANAGED_BUFFER_REMAP_KERNEL_OPENCL_CL__ */
/* end: sixtracklib/opencl/impl/managed_buffer_remap_kernel.cl */
