#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/impl/managed_buffer_remap_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>

    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/cuda/impl/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

extern __global__ void NS(ManagedBuffer_remap_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );

extern __global__ void NS(ManagedBuffer_remap_io_buffers_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );


__global__ void NS(ManagedBuffer_remap_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();

    size_t const thread_id_to_remap_buffers = ( size_t )0u;

    SIXTRL_ASSERT( NS(Cuda_get_total_num_threads_in_kernel)() >
        ( buf_size_t )0u );

    if( thread_id_to_remap_buffers == thread_id )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;
        int32_t success_flag       = ( int32_t )0u;

        if( ( buffer_begin != SIXTRL_NULLPTR ) &&
            ( 0 != NS(ManagedBuffer_remap)( buffer_begin, slot_size ) ) )
        {
            success_flag |= -2;
        }
        else if( buffer_begin != SIXTRL_NULLPTR )
        {
            success_flag |= -1;
        }

        if( ( success_flag == ( int32_t )0u ) &&
            ( ptr_success_flag != SIXTRL_NULLPTR ) )
        {
            #if ( defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) )
            atomicOr( ptr_success_flag, success_flag );
            #else  /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
            *ptr_success_flag |= success_flag;
            #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
        }
    }

    return;
}

__global__ void NS(ManagedBuffer_remap_io_buffers_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const total_num_threads = NS(Cuda_get_total_num_threads_in_kernel)();

    size_t const thread_id_to_remap_in_buffers = ( size_t )0u;

    size_t const thread_id_to_remap_out_buffers =
        ( total_num_threads > ( size_t )1u )
            ? ( thread_id_to_remap_in_buffers + ( size_t )1u )
            : ( thread_id_to_remap_in_buffers );

    if( thread_id <= thread_id_to_remap_out_buffers )
    {
        int32_t success_flag = ( int32_t )0u;
        buf_size_t const slot_size = ( buf_size_t )8u;

        if( thread_id == thread_id_to_remap_in_buffers )
        {
            if( ( in_buffer_begin != SIXTRL_NULLPTR ) &&
                ( in_buffer_begin != out_buffer_begin ) )
            {
                if( 0 != NS(ManagedBuffer_remap)(
                            in_buffer_begin, slot_size ) )
                {
                    success_flag |= -2;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( thread_id == thread_id_to_remap_out_buffers )
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

        if( ( success_flag == ( int32_t )0u ) &&
            ( ptr_success_flag != SIXTRL_NULLPTR ) )
        {
            #if ( defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) )
            atomicOr( ptr_success_flag, success_flag );
            #else  /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
            *ptr_success_flag |= success_flag;
            #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
        }
    }

    return;
}

/* end: sixtracklib/cuda/details/managed_buffer_remap_kernel.cu */
