#include "sixtracklib/cuda/impl/cuda_particles_kernel_c_wrapper.h"

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
    #include "sixtracklib/common/particles.h"

    #include "sixtracklib/cuda/impl/cuda_particles_kernel.cuh"
    #include "sixtracklib/cuda/impl/managed_buffer_remap_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__host__ int NS(Run_test_particles_copy_buffer_kernel_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const threads_per_block )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t const num_particle_blocks =
        NS(Particles_buffer_get_num_of_particle_blocks)( in_buffer );

    buf_size_t const total_num_particles =
        NS(Particles_buffer_get_total_num_of_particles)( in_buffer );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( in_buffer ) );

    if( ( num_particle_blocks > ZERO_SIZE ) &&
        ( total_num_particles > ZERO_SIZE ) &&
        ( num_blocks          > ZERO_SIZE ) &&
        ( threads_per_block   > ZERO_SIZE ) )
    {
        unsigned char const* in_buffer_begin = ( unsigned char const*
            )( uintptr_t )NS(Buffer_get_data_begin_addr)( in_buffer );

        unsigned char* out_buffer_begin = ( unsigned char* )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( out_buffer );

        buf_size_t const in_buffer_size  = NS(Buffer_get_size)( in_buffer  );
        buf_size_t const out_buffer_capacity = NS(Buffer_get_capacity)( out_buffer );

        if( ( in_buffer_begin  != SIXTRL_NULLPTR ) &&
            ( out_buffer_begin != SIXTRL_NULLPTR ) &&
            ( in_buffer_size    > ZERO_SIZE ) &&
            ( in_buffer_size   <= out_buffer_capacity ) )
        {
            int32_t success_flag = ( int32_t )0u;

            unsigned char* cuda_in_buffer   = SIXTRL_NULLPTR;
            unsigned char* cuda_out_buffer  = SIXTRL_NULLPTR;
            int32_t*       cuda_success_flag = SIXTRL_NULLPTR;

            if( cudaSuccess == cudaMalloc( ( void** )&cuda_in_buffer, in_buffer_size ) )
            {
                SIXTRL_ASSERT( cuda_in_buffer != SIXTRL_NULLPTR );
                success = 0;
            }
            else
            {
                success |= -32;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMemcpy( cuda_in_buffer,
                    in_buffer_begin, in_buffer_size, cudaMemcpyHostToDevice ) ) )
            {
                success |= -32;
            }

            if( success == 0 )
            {
                typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*  obj_iter_t;
                typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* ptr_particles_t;
                memcpy( out_buffer_begin, in_buffer_begin, in_buffer_size );

                if( 0 == NS(Buffer_remap)( out_buffer ) )
                {
                    NS(Particles_buffer_clear_particles)( out_buffer );
                }
                else
                {
                    success |= -64;
                }
            }

            SIXTRL_ASSERT( ( success != 0 ) ||
                ( NS(Buffer_get_size)( out_buffer ) == in_buffer_size ) );

            SIXTRL_ASSERT( NS(Particles_buffers_have_same_structure)(
                in_buffer, out_buffer ) );

            SIXTRL_ASSERT( NS(Particles_buffer_get_total_num_of_particles)(
                out_buffer ) == total_num_particles );

            if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                    ( void** )&cuda_out_buffer, in_buffer_size ) ) )
            {
                success |= -256;
            }

            out_buffer_begin = ( unsigned char* )( uintptr_t
                )NS(Buffer_get_data_begin_addr)( out_buffer );

            if( ( success == 0 ) &&
                ( cudaSuccess != cudaMemcpy( cuda_out_buffer, out_buffer_begin,
                    in_buffer_size, cudaMemcpyHostToDevice ) ) )
            {
                success |= -512;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMalloc( (
                void** )&cuda_success_flag, sizeof( success_flag ) ) ) )
            {
                success |= -1024;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMemcpy(
                cuda_success_flag, &success_flag, sizeof( success_flag ),
                    cudaMemcpyHostToDevice ) ) )
            {
                success |= -2048;
            }

            if( success == 0 )
            {
                dim3 grid_dim;
                dim3 block_dim;

                success = -4096;

                grid_dim.x  = ( num_blocks >= ( buf_size_t )2u ) ? 2u : num_blocks;
                grid_dim.y  = 1;
                grid_dim.z  = 1;

                block_dim.x = 1;
                block_dim.y = 1;
                block_dim.z = 1;

                NS(ManagedBuffer_remap_io_buffers_kernel_cuda)<<<
                    grid_dim, block_dim >>>( cuda_in_buffer,
                        cuda_out_buffer, cuda_success_flag );

                if( ( cudaSuccess == cudaDeviceSynchronize() ) &&
                    ( cudaSuccess == cudaMemcpy( &success_flag, cuda_success_flag,
                        sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
                {
                    success = success_flag;
                }
            }

            if( success == 0 )
            {
                dim3 grid_dim;
                dim3 block_dim;

                success = -8192;

                grid_dim.x = num_blocks;
                grid_dim.y = 1;
                grid_dim.z = 1;

                block_dim.x = threads_per_block;
                block_dim.y = 1;
                block_dim.z = 1;

                NS(Particles_copy_buffer_kernel_cuda)<<< grid_dim, block_dim >>>(
                    cuda_in_buffer, cuda_out_buffer, cuda_success_flag );

                if( ( cudaSuccess == cudaDeviceSynchronize() ) &&
                    ( cudaSuccess == cudaMemcpy( &success_flag, cuda_success_flag,
                        sizeof( success_flag ), cudaMemcpyDeviceToHost ) ) )
                {
                    success |= ( int )success_flag;

                    if( ( success_flag == 0 ) && ( cudaSuccess == cudaMemcpy(
                        out_buffer_begin, cuda_out_buffer, in_buffer_size,
                            cudaMemcpyDeviceToHost ) ) )
                    {
                        if( 0 == NS(Buffer_remap)( out_buffer ) )
                        {
                            success = 0;
                        }
                        else
                        {
                            success |= -16384;
                        }
                    }
                }
            }

            if( ( success == 0 ) && ( 0 != NS(Buffer_remap)( out_buffer ) ) )
            {
                success |= -32768;
            }

            if( ( ( cuda_out_buffer != SIXTRL_NULLPTR ) &&
                  ( cudaSuccess != cudaFree( cuda_out_buffer ) ) ) ||
                ( ( cuda_in_buffer  != SIXTRL_NULLPTR ) &&
                  ( cudaSuccess != cudaFree( cuda_in_buffer ) ) ) ||
                ( ( cuda_success_flag != SIXTRL_NULLPTR ) &&
                  ( cudaSuccess != cudaFree( cuda_success_flag ) ) ) )
            {
                success |= -65536;
            }
        }
    }

    return success;
}

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

__host__ int NS(Run_test_particles_copy_buffer_kernel_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer) const* SIXTRL_RESTRICT in_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer )
{
    int success = -131072;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    buf_size_t total_num_particles  = ZERO_SIZE;

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

    total_num_particles =
        NS(Particles_buffer_get_total_num_of_particles)( in_buffer );

    if( cudaSuccess == cudaOccupancyMaxPotentialBlockSize( &min_grid_size,
            &block_size, NS(Particles_copy_buffer_kernel_cuda), 0u, 0u ) )
    {
        success = 0;
    }

    if( ( success == 0 ) && ( block_size > 0 ) )
    {
        grid_size = ( total_num_particles + block_size - 1 ) / block_size;

    }

    #if !defined( _NDEBUG )

    if( ( success == 0 ) && ( cudaSuccess !=
          cudaOccupancyMaxActiveBlocksPerMultiprocessor( &max_active_blocks,
            NS(Particles_copy_buffer_kernel_cuda), block_size, 0u ) ) )
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

            printf( "DEBUG :: Launch kernel NS(Particles_copy_buffer_kernel_cuda) "
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
        success = NS(Run_test_particles_copy_buffer_kernel_on_cuda_grid)(
            in_buffer, out_buffer, grid_size, block_size );
    }

    return success;
}

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

/* end: tests/sixtracklib/cuda/details/cuda_particles_kernel_c_wrapper.cu */
