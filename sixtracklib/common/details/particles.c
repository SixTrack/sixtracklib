#include "sixtracklib/common/particles.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"

#include "sixtracklib/common/alignment.h"
#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/blocks_container.h"


extern int NS(ParticlesContainer_add_particles)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer, 
    NS(Particles)* SIXTRL_RESTRICT particle_block,
    NS(block_num_elements_t) const num_of_particles );

extern int NS(ParticlesContainer_add_blocks_of_particles)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(Particles)* SIXTRL_RESTRICT particle_blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const* SIXTRL_RESTRICT num_of_particles_vec );

/* ------------------------------------------------------------------------- */

int NS(ParticlesContainer_add_particles)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer, 
    NS(Particles)* SIXTRL_RESTRICT particle_block,
    NS(block_num_elements_t) const num_of_particles )
{
    int success = -1;
    
    if( ( particles_buffer != 0 ) && 
        ( NS(ParticlesContainer_get_block_capacity)( particles_buffer ) >
          NS(ParticlesContainer_get_num_of_blocks)( particles_buffer ) ) )
    {
        static NS(block_size_t) const INFO_SIZE = sizeof( NS(BlockInfo ) );
        
        NS(block_alignment_t) const info_align = 
            NS(ParticlesContainer_get_info_alignment)( particles_buffer );
            
        NS(block_alignment_t) const data_align =
            NS(ParticlesContainer_get_data_alignment)( particles_buffer );
        
        NS(MemPool) rollback_info_store = particles_buffer->info_store;
        NS(MemPool) rollback_data_store = particles_buffer->data_store;
        
        NS(AllocResult) info_result = NS(MemPool_append_aligned)(
            &particles_buffer->info_store, INFO_SIZE, info_align );
        
        NS(block_size_t) const mem_offset = NS(MemPool_get_next_begin_offset)( 
            &particles_buffer->data_store, data_align );
        
        NS(block_size_t) const max_num_bytes_on_mem = 
            NS(MemPool_get_capacity)( &particles_buffer->data_store );
        
        if( ( NS(AllocResult_valid)( &info_result ) ) &&
            ( max_num_bytes_on_mem > mem_offset ) )
        {
            NS(BlockType) const type_id = NS(BLOCK_TYPE_PARTICLE);
            
            NS(BlockInfo)* block_info = 
                ( NS(BlockInfo)* )NS(AllocResult_get_pointer)( &info_result );
            
            unsigned char* data_mem_begin = 
                NS(ParticlesContainer_get_ptr_data_begin)( particles_buffer );
                
            NS(BlockInfo_set_mem_offset)( block_info, mem_offset );
            NS(BlockInfo_set_type_id)( block_info, type_id );
            NS(BlockInfo_set_common_alignment)( block_info, data_align );
            NS(BlockInfo_set_num_elements)( block_info, 
                                            ( NS(block_num_elements_t) )1u );
            
            NS(Particles_preset)( particle_block );
            NS(Particles_set_num_particles)( particle_block, num_of_particles );
            NS(Particles_set_type_id)( particle_block, NS(BLOCK_TYPE_PARTICLE ) );
            
            if( 0 == NS(Particles_map_to_memory_for_writing_aligned)( 
                particle_block, block_info, data_mem_begin, 
                    max_num_bytes_on_mem ) )
            {
                ++particles_buffer->num_blocks;
                
                NS(MemPool_increment_size)( 
                    &particles_buffer->data_store, 
                    NS(BlockInfo_get_mem_offset)( block_info ) + 
                    NS(BlockInfo_get_num_of_bytes)( block_info ) );
                
                success = 0;
                
                return success;
            }
        }
        
        /* if we are here, something went wrong -> rollback and return an 
         * invalid drift! */
        
        SIXTRL_ASSERT( NS(Particles_get_type_id)( particle_block ) == 
                       NS(BLOCK_TYPE_INVALID ) );
        
        particles_buffer->info_store = rollback_info_store;
        particles_buffer->data_store = rollback_data_store;
    }
    
    NS(Particles_preset)( particle_block );
    return success;
}

int NS(ParticlesContainer_add_blocks_of_particles)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(Particles)* SIXTRL_RESTRICT particle_blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const* SIXTRL_RESTRICT num_of_particles_vec )
{
    int success = -1;
    
    if( ( particles_buffer != 0 ) && ( particle_blocks != 0 ) &&
        ( num_of_particles_vec != 0 ) && 
        ( num_of_blocks > ( NS(block_size_t) )0u ) &&
        ( NS(ParticlesContainer_get_block_capacity)( particles_buffer ) >=
          NS(ParticlesContainer_get_num_of_blocks)( 
            particles_buffer ) + num_of_blocks ) )
    {
        NS(block_size_t) ii = ( NS(block_size_t) )0u;
        
        success = 0;
        
        for( ; ii < num_of_blocks ; ++ii )
        {
            int const result_ii = NS(ParticlesContainer_add_particles)(
                particles_buffer, &particle_blocks[ ii ], 
                    num_of_particles_vec[ ii ] );
            
            if( result_ii != 0 )
            {
                success = -1;
                break;
            }            
        }
    }
        
    return success;
}

