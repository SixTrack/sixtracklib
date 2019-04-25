#include "sixtracklib/testlib/cuda/cuda_buffer_generic_obj_kernel_c_wrapper.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>

    #if !defined( NDEBUG )
        #include <stdio.h>
    #endif /* !defined( NDEBUG ) */

    #include <cuda_runtime.h>
    #include <cuda_occupancy.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/beam_elements.h"

    #include "sixtracklib/testlib/cuda/kernels/cuda_buffer_generic_obj_kernel.cuh"
    #include "sixtracklib/testlib/cuda/kernels/cuda_beam_elements_kernel.cuh"
    #include "sixtracklib/cuda/kernels/managed_buffer_remap_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__host__ int NS(Run_test_buffer_generic_obj_kernel_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const threads_per_block )
{
    typedef NS(buffer_size_t) buf_size_t;

    int success = -16;

    buf_size_t const slot_size    = NS(Buffer_get_slot_size)( in_buffer );
    size_t const orig_buffer_size = NS(Buffer_get_size)( in_buffer );
    size_t const copy_buffer_size = NS(Buffer_get_size)( out_buffer );

    unsigned char const* orig_buffer_begin =
        NS(Buffer_get_const_data_begin)( in_buffer );

    unsigned char* copy_buffer_begin = NS(Buffer_get_data_begin)( out_buffer );

    if( ( orig_buffer_begin != SIXTRL_NULLPTR ) &&
        ( copy_buffer_begin != SIXTRL_NULLPTR ) &&
        ( orig_buffer_size > slot_size ) &&
        ( orig_buffer_size == copy_buffer_size ) )
    {
        int32_t success_flag = 0;

        unsigned char* cuda_orig_begin = SIXTRL_NULLPTR;
        unsigned char* cuda_copy_begin = SIXTRL_NULLPTR;
        int32_t* cuda_success_flag     = SIXTRL_NULLPTR;

        SIXTRL_ASSERT( orig_buffer_size == copy_buffer_size );

        SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( in_buffer ) ==
                       NS(Buffer_get_num_of_objects)( out_buffer ) );

        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( in_buffer  ) );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );

        if( cudaSuccess == cudaMalloc(
            ( void** )&cuda_orig_begin, orig_buffer_size ) )
        {
            success = 0;
        }

        if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                ( void** )&cuda_copy_begin, copy_buffer_size ) ) )
        {
            success |= -32;
        }

        if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                ( void** )&cuda_success_flag, sizeof( success_flag ) ) ) )
        {
            success |= -64;
        }

        SIXTRL_ASSERT( ( success != 0 ) ||
                       ( cuda_orig_begin   != SIXTRL_NULLPTR ) &&
                       ( cuda_copy_begin   != SIXTRL_NULLPTR ) &&
                       ( cuda_success_flag != SIXTRL_NULLPTR ) );

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_orig_begin, orig_buffer_begin,
                orig_buffer_size, cudaMemcpyHostToDevice ) ) )
        {
            success |= -128;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_copy_begin, copy_buffer_begin,
                copy_buffer_size, cudaMemcpyHostToDevice ) ) )
        {
            success |= -256;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( cuda_success_flag, &success_flag,
                sizeof( success_flag ), cudaMemcpyHostToDevice ) ) )
        {
            success |= -512;
        }


        if( success == 0 )
        {
            dim3 grid_dim;
            dim3 block_dim;

            grid_dim.x  = ( num_blocks >= ( buf_size_t )2u ) ? 2u : num_blocks;
            grid_dim.y  = 1;
            grid_dim.z  = 1;

            block_dim.x = 1;
            block_dim.y = 1;
            block_dim.z = 1;

            NS(Remap_original_buffer_kernel_cuda)<<< grid_dim, block_dim >>>(
                cuda_orig_begin, cuda_copy_begin, cuda_success_flag );

            if( cudaSuccess != cudaDeviceSynchronize() )
            {
                success |= -1024;
            }
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( &success_flag, cuda_success_flag,
                sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
        {
            success |= -2048;
        }

        if( success == 0 )
        {
            success |= ( int )success_flag;
        }

        if( success == 0 )
        {
            dim3 grid_dim;
            dim3 block_dim;

            grid_dim.x = num_blocks;
            grid_dim.y = 1;
            grid_dim.z = 1;

            block_dim.x = threads_per_block;
            block_dim.y = 1;
            block_dim.z = 1;

            NS(Copy_original_buffer_kernel_cuda)<<< grid_dim, block_dim >>>(
                cuda_orig_begin, cuda_copy_begin, cuda_success_flag );

            if( cudaSuccess != cudaDeviceSynchronize() )
            {
                success |= -4096;
            }
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( &success_flag, cuda_success_flag,
                sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
        {
            success |= -8192;
        }

        if( success == 0 )
        {
            success |= ( int )success_flag;
        }

        if( ( success == 0 ) &&
            ( cudaSuccess != cudaMemcpy( copy_buffer_begin, cuda_copy_begin,
                copy_buffer_size, cudaMemcpyDeviceToHost ) ) )
        {
            success |= -16384;
        }

        if( ( success == 0 ) && ( 0 != NS(Buffer_remap)( out_buffer ) ) )
        {
            success |= -32768;
        }

        if( ( ( cuda_orig_begin != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_orig_begin ) ) ) ||
            ( ( cuda_copy_begin != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_copy_begin ) ) ) ||
            ( ( cuda_success_flag != SIXTRL_NULLPTR ) &&
              ( cudaSuccess != cudaFree( cuda_success_flag ) ) ) )
        {
            success |= -65536;
        }
    }

    return success;
}

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

__host__ int NS(Run_test_buffer_generic_obj_kernel_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer )
{
    int success = -131072;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t num_beam_elements = ZERO_SIZE;

    /* Cf. NVIDIA DevBlogs, July 17th, 2014
    *  "CUDA Pro Tip: Occupancy API Simplifies Launch Configuration"
    *  by Mark Harris
    *  (shortened URL: https://preview.tinyurl.com/yce9ntnt )
    *  for reference! */

    int block_size        = 0;
    int min_grid_size     = 0;
    int grid_size         = 0;
    int max_active_blocks = 0;

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( in_buffer  ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );

    num_beam_elements = NS(Buffer_get_num_of_objects)( in_buffer );

    if( cudaSuccess == cudaOccupancyMaxPotentialBlockSize( &min_grid_size,
            &block_size, NS(Copy_original_buffer_kernel_cuda), 0u, 0u ) )
    {
        success = 0;
    }

    if( ( success == 0 ) && ( block_size > 0 ) )
    {
        grid_size = ( num_beam_elements + block_size - 1 ) / block_size;

    }

    #if !defined( _NDEBUG )

    if( ( success == 0 ) && ( cudaSuccess !=
          cudaOccupancyMaxActiveBlocksPerMultiprocessor( &max_active_blocks,
            NS(Copy_original_buffer_kernel_cuda), block_size, 0u ) ) )
    {
        success |= -262144;
    }

    if( success == 0 )
    {
        int device_id = 0;
        cudaDeviceProp device_info;

        if( ( cudaSuccess == cudaGetDevice( &device_id ) ) &&
            ( cudaSuccess == cudaGetDeviceProperties( &device_info, device_id ) ) )
        {
            double const occupancy =
                ( double )( ( max_active_blocks * block_size ) / device_info.warpSize ) /
                ( double )( device_info.maxThreadsPerMultiProcessor *
                            device_info.warpSize );

            printf( "DEBUG :: Launch kernel "
                    "NS(Copy_original_buffer_kernel_cuda) "
                    "with a block_size of %d; Theoretical occupancy = %f\r\n",
                    block_size, occupancy );
        }
        else
        {
            success |= -524288;
        }
    }

    #endif /* !defined( NDEBUG ) */

    if( success == 0 )
    {
        success = NS(Run_test_buffer_generic_obj_kernel_on_cuda_grid)(
            in_buffer, out_buffer, grid_size, block_size );
    }

    return success;
}

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

/* end: tests/sixtracklib/cuda/cuda_buffer_generic_obj_kernel_c_wrapper.h */
