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


extern int NS(ParticlesContainer_add_blocks_of_particles)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(Particles)* SIXTRL_RESTRICT particle_blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const* SIXTRL_RESTRICT num_of_particles_vec );

/* ------------------------------------------------------------------------- */

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

