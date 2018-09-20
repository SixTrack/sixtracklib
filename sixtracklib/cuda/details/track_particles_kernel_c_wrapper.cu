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
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"

    #include "sixtracklib/cuda/impl/track_particles_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__host__ int NS(Track_particles_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const threads_per_block );

#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

__host__ int NS(Track_particles_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns );

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

/* ------------------------------------------------------------------------- */

__host__ int NS(Track_particles_on_cuda_grid)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns,
    NS(buffer_size_t) const num_blocks,
    NS(buffer_size_t) const threads_per_block )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( in_particles ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( beam_elements ) );

    if( ( NS(Buffer_get_num_of_objects)( in_particles )  > ZERO_SIZE ) &&
        ( NS(Buffer_get_num_of_objects)( beam_elements ) > ZERO_SIZE ) &&
        ( num_turns           > ZERO_SIZE ) &&
        ( num_blocks          > ZERO_SIZE ) &&
        ( threads_per_block   > ZERO_SIZE ) )
    {
        unsigned char const* in_particles_begin = ( unsigned char const*
            )( uintptr_t )NS(Buffer_get_data_begin_addr)( in_particles );

        unsigned char const* beam_elements_begin = (unsigned char const*
            )( uintptr_t )NS(Buffer_get_data_begin_addr)( beam_elements );

        unsigned char* result_particles_begin = ( unsigned char*
            )( uintptr_t )NS(Buffer_get_data_begin_addr)( result_particles );

        buf_size_t const beam_elements_buffer_size =
            NS(Buffer_get_size)( beam_elements );

        buf_size_t const particles_buffer_size =
            NS(Buffer_get_size)( in_particles );

        buf_size_t const result_particles_buffer_capacity =
            NS(Buffer_get_capacity)( result_particles );

        if( ( in_particles_begin     != SIXTRL_NULLPTR ) &&
            ( beam_elements_begin    != SIXTRL_NULLPTR ) &&
            ( result_particles_begin != SIXTRL_NULLPTR ) &&
            ( beam_elements_buffer_size > ZERO_SIZE ) &&
            ( particles_buffer_size     > ZERO_SIZE ) &&
            ( particles_buffer_size <= result_particles_buffer_capacity ) )
        {
            int32_t success_flag = ( int32_t )0u;

            unsigned char* cuda_particles_buffer     = SIXTRL_NULLPTR;
            unsigned char* cuda_beam_elements_buffer = SIXTRL_NULLPTR;
            int32_t*       cuda_succes_flag_buffer   = SIXTRL_NULLPTR;

            if( cudaSuccess == cudaMalloc( ( void** )&cuda_particles_buffer,
                    particles_buffer_size ) )
            {
                SIXTRL_ASSERT( cuda_particles_buffer != SIXTRL_NULLPTR );
                success = 0;
            }
            else
            {
                success |= -16;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMemcpy(
                cuda_particles_buffer, in_particles_begin,
                    particles_buffer_size, cudaMemcpyHostToDevice ) ) )
            {
                success |= -16;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMalloc(
                    ( void** )&cuda_beam_elements_buffer,
                        beam_elements_buffer_size ) ) )
            {
                success |= -32;
            }

            if( ( success == 0 ) &&
                ( cudaSuccess != cudaMemcpy( cuda_beam_elements_buffer,
                    beam_elements_begin, beam_elements_buffer_size,
                        cudaMemcpyHostToDevice ) ) )
            {
                success |= -32;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMalloc( (
                void** )&cuda_succes_flag_buffer, sizeof( success_flag ) ) ) )
            {
                success |= -64;
            }

            if( ( success == 0 ) && ( cudaSuccess != cudaMemcpy(
                cuda_succes_flag_buffer, &success_flag, sizeof( success_flag ),
                    cudaMemcpyHostToDevice ) ) )
            {
                success |= -64;
            }

            if( success == 0 )
            {
                dim3 grid_dim;
                dim3 block_dim;

                success = -128;

                grid_dim.x  = ( num_blocks >= ( buf_size_t )2u ) ? 2u : num_blocks;
                grid_dim.y  = 1;
                grid_dim.z  = 1;

                block_dim.x = 1;
                block_dim.y = 1;
                block_dim.z = 1;

                NS(Remap_particles_beam_elements_buffers_kernel_cuda)<<<
                    grid_dim, block_dim >>>( cuda_particles_buffer,
                        cuda_beam_elements_buffer, cuda_succes_flag_buffer );

                if( ( cudaSuccess == cudaDeviceSynchronize() ) &&
                    ( cudaSuccess == cudaMemcpy( &success_flag,
                        cuda_succes_flag_buffer, sizeof( success_flag ),
                            cudaMemcpyDeviceToHost ) ) )
                {
                    success = success_flag;
                }
            }

            if( success == 0 )
            {
                dim3 grid_dim;
                dim3 block_dim;

                success = -256;

                grid_dim.x = num_blocks;
                grid_dim.y = 1;
                grid_dim.z = 1;

                block_dim.x = threads_per_block;
                block_dim.y = 1;
                block_dim.z = 1;

                NS(Track_particles_beam_elements_kernel_cuda)<<<
                    grid_dim, block_dim >>>( cuda_particles_buffer,
                    cuda_beam_elements_buffer, num_turns,
                        cuda_succes_flag_buffer );

                if( ( cudaSuccess == cudaDeviceSynchronize() ) &&
                    ( cudaSuccess == cudaMemcpy( &success_flag,
                        cuda_succes_flag_buffer, sizeof( success_flag ),
                            cudaMemcpyDeviceToHost ) ) )
                {
                    if( ( success_flag == 0 ) && ( cudaSuccess == cudaMemcpy(
                        result_particles_begin, cuda_particles_buffer,
                            particles_buffer_size, cudaMemcpyDeviceToHost ) ) )
                    {
                        if( 0 != NS(Buffer_remap)( result_particles ) )
                        {
                            success |= -512;
                        }
                    }
                }
            }

            if( cuda_particles_buffer != SIXTRL_NULLPTR )
            {
                if( cudaSuccess != cudaFree( cuda_particles_buffer ) )
                {
                    success |= -1024;
                }

                cuda_particles_buffer = SIXTRL_NULLPTR;
            }

            if( cuda_beam_elements_buffer != SIXTRL_NULLPTR )
            {
                if( cudaSuccess != cudaFree( cuda_beam_elements_buffer ) )
                {
                    success |= -2048;
                }

                cuda_beam_elements_buffer = SIXTRL_NULLPTR;
            }

            if( cuda_succes_flag_buffer != SIXTRL_NULLPTR )
            {
                if( cudaSuccess != cudaFree( cuda_succes_flag_buffer ) )
                {
                    success |= -4096;
                }

                cuda_succes_flag_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    return success;
}


#if defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 )

__host__ int NS(Track_particles_on_cuda)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* result_particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const in_particles,
    NS(buffer_size_t) const num_turns )
{
    int success = -8192;

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

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( in_particles  ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( beam_elements ) );

    total_num_particles =
        NS(Particles_buffer_get_total_num_of_particles)( in_particles );

    if( cudaSuccess == cudaOccupancyMaxPotentialBlockSize( &min_grid_size,
            &block_size, NS(Track_particles_beam_elements_kernel_cuda), 0u, 0u ) )
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
            NS(Track_particles_beam_elements_kernel_cuda), block_size, 0u ) ) )
    {
        success |= -16384;
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

            printf( "DEBUG :: Launch kernel NS(Track_particles_beam_elements_kernel_cuda) "
                    "with a block_size of %d; Theoretical occupancy = %f\r\n",
                    block_size, occupancy );
        }
        else
        {
            success |= -16384;
        }
    }

    #endif /* !defined( NDEBUG ) */

    if( success == 0 )
    {
        success = NS(Track_particles_on_cuda_grid)(
            result_particles, beam_elements, in_particles,
                num_turns, grid_size, block_size );
    }

    return success;
}

#endif /* defined( CUDART_VERSION ) && ( CUDART_VERSION >= 6050 ) */

/* end: sixtracklib/cuda/details/track_particles_kernel_c_wrapper.cu */
