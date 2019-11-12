#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/managed_buffer_remap.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(ManagedBuffer_remap_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size )
{
    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( size_t )0 )
    {
        NS(ManagedBuffer_remap)( buffer_begin, slot_size );
    }
}

__global__ void NS(ManagedBuffer_remap_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef NS(arch_status_t) status_t;
    typedef NS(buffer_size_t) buf_size_t;

    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( buf_size_t )0u )
    {
        NS(arch_debugging_t) dbg = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;
        status_t const status = NS(ManagedBuffer_remap)( buffer, slot_size );

        if( ptr_dbg_register != SIXTRL_NULLPTR )
        {
            if( status != SIXTRL_ARCH_STATUS_SUCCESS )
            {
                if( buffer == SIXTRL_NULLPTR )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

                if( slot_size == ( buf_size_t )0u )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

                if( NS(ManagedBuffer_needs_remapping)( buffer, slot_size ) )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );
            }

            *ptr_dbg_register = ::NS(DebugReg_store_arch_status)( dbg, status );
        }
    }
}

__global__ void NS(ManagedBuffer_needs_remapping_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    NS(arch_debugging_t) const needs_remapping_true,
    NS(arch_debugging_t) const needs_remapping_false,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_result )
{
    typedef NS(buffer_size_t) buf_size_t;

    if( NS(Cuda_get_1d_thread_id_in_kernel)() == ( buf_size_t )0u )
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

/* end: sixtracklib/cuda/kernels/managed_buffer_remap.cu */
