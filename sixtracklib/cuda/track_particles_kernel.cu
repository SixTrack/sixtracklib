#include "sixtracklib/cuda/track_particles_kernel.cuh"

#include <stdio.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/impl/particles_api.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/beam_elements_type.h"
#include "sixtracklib/common/impl/beam_elements_api.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/impl/track_api.h"

__global__ void Track_remap_serialized_blocks_buffer(
    unsigned char* __restrict__ particles_data_buffer,
    unsigned char* __restrict__ beam_elements_data_buffer,
    unsigned char* __restrict__ elem_by_elem_data_buffer, 
    int64_t* __restrict__ success_flag )
{
    int const global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int const total_num_threads = blockDim.x * gridDim.x;
    
    int const gid_to_remap_particles     = 0;
    int const gid_to_remap_beam_elements = ( total_num_threads > 1 )
        ? 1 : gid_to_remap_particles;
    int const gid_to_remap_elem_by_elem  = ( total_num_threads > 2 )
        ? 2 : gid_to_remap_beam_elements;
    
    if( global_id <= gid_to_remap_elem_by_elem )
    {
        int ret = 0;
    
        NS(Blocks) particles_buffer;
        NS(Blocks) beam_elements;
        NS(Blocks) elem_by_elem_buffer;
        
        NS(Blocks_preset)( &particles_buffer );
        NS(Blocks_preset)( &beam_elements );
        NS(Blocks_preset)( &elem_by_elem_buffer );
        
        if( global_id == gid_to_remap_particles )
        {
            ret |= NS(Blocks_unserialize)( &particles_buffer, 
                        particles_data_buffer );        
            
            if( ( ret != 0 ) && ( success_flag != NULL ) )
            {
                *success_flag = -1;
            }        
        }
        
        if( ( ret == 0 ) && ( global_id == gid_to_remap_beam_elements ) )
        {        
            ret |= NS(Blocks_unserialize)( &beam_elements, 
                        beam_elements_data_buffer );
            
            if( ( ret != 0 ) && ( success_flag != NULL ) )
            {
                *success_flag = -1;
            }
        }
        
        if( ( ret == 0 ) && ( global_id == gid_to_remap_elem_by_elem ) )
        {
            ret |= NS(Blocks_unserialize)( &elem_by_elem_buffer, 
                        elem_by_elem_data_buffer );
            
            if( ( ret != 0 ) && ( elem_by_elem_data_buffer != NULL ) )
            {
                uint64_t const* header = 
                    ( uint64_t const* )elem_by_elem_data_buffer;
                    
                if( ( header[ 0 ] != ( uint64_t )0u ) || 
                    ( header[ 1 ] != ( uint64_t )0u ) ||
                    ( header[ 2 ] != ( uint64_t )0u ) ||
                    ( header[ 3 ] != ( uint64_t )0u ) )
                {
                    if( success_flag != NULL )
                    {
                        *success_flag = -1;
                    }
                }
            }
            else if( success_flag != NULL )
            {
                *success_flag = -1;
            }
        }
    }

    return;
}

__global__ void Track_particles_kernel_cuda(
    SIXTRL_UINT64_T const num_of_turns,
    unsigned char* __restrict__ particles_data_buffer,
    unsigned char* __restrict__ beam_elements_data_buffer,
    unsigned char* __restrict__ elem_by_elem_data_buffer, 
    int64_t* __restrict__ success_flag )
{
    int const global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ret = 0;
    bool use_elem_by_elem_buffer = false;
    
    NS(block_size_t) num_particle_blocks     = 0;
    NS(block_size_t) num_beam_elements       = 0;
    NS(block_size_t) num_elem_by_elem_blocks = 0;
    
    NS(Blocks) particles_buffer;
    NS(Blocks) beam_elements;
    NS(Blocks) elem_by_elem_buffer;
    
    NS(Blocks_preset)( &particles_buffer );
    NS(Blocks_preset)( &beam_elements );
    NS(Blocks_preset)( &elem_by_elem_buffer );
        
    ret  = NS(Blocks_unserialize)( 
        &particles_buffer, particles_data_buffer );
        
    ret |= NS(Blocks_unserialize)( 
        &beam_elements, beam_elements_data_buffer );
    
    num_particle_blocks = NS(Blocks_get_num_of_blocks)( &particles_buffer );
    num_beam_elements   = NS(Blocks_get_num_of_blocks)( &beam_elements );
    
    if( 0 == NS(Blocks_unserialize)( 
            &elem_by_elem_buffer, elem_by_elem_data_buffer ) )
    {
        NS(block_size_t) required_num_elem_by_elem_blocks = 
            num_of_turns * num_beam_elements * num_particle_blocks;
    
        num_elem_by_elem_blocks = 
            NS(Blocks_get_num_of_blocks)( &elem_by_elem_buffer );
    
        use_elem_by_elem_buffer = ( 
            num_elem_by_elem_blocks >= required_num_elem_by_elem_blocks );
    }
    
    if( ( ret != 0 ) && ( success_flag != NULL ) )
    {
        *success_flag = -1;
        printf( "global_id: %5d :: error in unserializing!\r\n", global_id );
    }

    __syncthreads();
    
    if( ( ret == 0 ) && ( num_beam_elements != 0u ) && 
        ( num_of_turns != 0u ) && ( num_particle_blocks == 1u ) )
    {
        NS(BlockInfo)* particle_blk_it = 
            NS(Blocks_get_block_infos_begin)( &particles_buffer );
            
        NS(Particles)* particles = NS(Blocks_get_particles)( particle_blk_it );            
        int const num_particles = NS(Particles_get_num_particles)( particles );
            
        if( ( particles != NULL ) && ( num_particles > global_id ) )
        {
            int const stride = blockDim.x * gridDim.x;            
        
            if( use_elem_by_elem_buffer )
            {
                NS(BlockInfo)* io_block_it = 
                    NS(Blocks_get_block_infos_begin)( &elem_by_elem_buffer );
                    
                for( uint64_t ii = 0u ; ii < num_of_turns ; ++ii, 
                    io_block_it = io_block_it + num_beam_elements )
                {
                    for( int jj = global_id; jj < num_particles; jj += stride )
                    {
                        ret |= NS(Track_beam_elements_particle)(
                            particles, jj, &beam_elements, io_block_it );
                    }
                }
            }
            else
            {
                for( uint64_t ii = 0u ; ii < num_of_turns ; ++ii )
                {
                    for( int jj = global_id; jj < num_particles; jj += stride )
                    {
                        ret |= NS(Track_beam_elements_particle)(
                            particles, jj, &beam_elements, NULL );
                    }
                }
            }
            
            if( ( ret != 0 ) && ( success_flag != NULL ) )
            {
                *success_flag = -1;
            }
        }
        else if( ( particles == NULL ) && ( num_particles > 0 ) && 
                 ( success_flag != NULL ) )
        {
            *success_flag = -1;
        }
    }
    else if( success_flag != NULL )
    {
        *success_flag = -1;
    }
       
    return;  
}

/* end sixtracklib/cuda/track_particles_kernel.cu */
