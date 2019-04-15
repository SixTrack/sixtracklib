#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/managed_buffer_remap.cuh"
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
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(ManagedBuffer_remap_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    uint64_t const slot_size )
{
    size_t const thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const thread_id_to_remap = ( size_t )0u;

    SIXTRL_ASSERT( NS(Cuda_get_total_num_threads_in_kernel)() >
        ( NS(buffer_size_t) )0u );

    if( thread_id_to_remap == thread_id )
    {
        int32_t const ret = NS(ManagedBuffer_remap)( buffer_begin, slot_size );
        SIXTRL_ASSERT( ret == 0 );
        ( void )ret;
    }

    return;
}

__global__ void NS(ManagedBuffer_remap_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    uint64_t const slot_size,
    SIXTRL_BUFFER_DATAPTR_DEC uint32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t ZERO = ( buf_size_t )0u;
    uint32_t success_flag = ( uint32_t )0x80000000;

    size_t const thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const thread_id_to_remap = ( size_t )0u;

    if( NS(Cuda_get_total_num_threads_in_kernel)() > ZERO )
    {
        if( thread_id_to_remap == thread_id )
        {
            if( ( buffer != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) )
            {
                int32_t const ret = NS(ManagedBuffer_remap)( buffer, slot_size );

                if( ( ret == ( int32_t )0 ) &&
                    ( !NS(ManagedBuffer_needs_remapping)( buffer, slot_size ) )
                  )
                {
                    success_flag = ( uint32_t )0u;
                }
                else if( ret != ( int32_t )0 )
                {
                    success_flag |= ( ret >= 0 ) ? ret : -ret;
                }
                else
                {
                    success_flag |= ( uint32_t )0x10000000;
                }
            }
            else if( buffer != SIXTRL_NULLPTR )
            {
                success_flag = ( uint32_t )0x20000000;
            }
            else
            {
                success_flag = ( uint32_t )0x40000000;
            }
        }
        else
        {
            success_flag = ( uint32_t )0;
        }
    }

    if( ( success_flag != ( uint32_t )0u ) &&
        ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        #if ( defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) )
        atomicOr( ptr_success_flag, success_flag );
        #else  /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
        *ptr_success_flag |= success_flag;
        #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
    }

}

/* end: sixtracklib/cuda/kernels/managed_buffer_remap.cu */
