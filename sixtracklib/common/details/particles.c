#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"

extern SIXTRL_GLOBAL_DEC NS(Particles)* 
NS(Blocks_add_particles)( NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_particles );


SIXTRL_GLOBAL_DEC NS(Particles)* NS(Blocks_add_particles)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_particles )
{
    SIXTRL_GLOBAL_DEC NS(Particles)* ptr_particles = 0;
    
    SIXTRL_STATIC NS(block_size_t) const NUM_ATTR_DATA_POINTERS = 20u;
    SIXTRL_STATIC NS(block_size_t) const REAL_SIZE = sizeof( SIXTRL_REAL_T  );
    SIXTRL_STATIC NS(block_size_t) const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    
    NS(block_size_t) const data_attr_sizes[] =
    {
        REAL_SIZE, REAL_SIZE, REAL_SIZE, REAL_SIZE,
        REAL_SIZE, REAL_SIZE, REAL_SIZE, REAL_SIZE,
        REAL_SIZE, REAL_SIZE, REAL_SIZE, REAL_SIZE,
        REAL_SIZE, REAL_SIZE, REAL_SIZE, REAL_SIZE,
        I64_SIZE,  I64_SIZE,  I64_SIZE,  I64_SIZE
    };
    
    if( ( blocks != 0 ) && 
        ( num_of_particles > ( NS(block_num_elements_t) )0u ) )
    {
        NS(Particles) particles;
        
        NS(block_size_t) const num = ( NS(block_size_t) )num_of_particles;
    
        NS(block_size_t) const data_attr_offsets[] = 
        {
            ( NS(block_size_t) )offsetof( NS(Particles), q0 ),
            ( NS(block_size_t) )offsetof( NS(Particles), mass0 ),
            ( NS(block_size_t) )offsetof( NS(Particles), beta0 ),
            ( NS(block_size_t) )offsetof( NS(Particles), gamma0 ),
            ( NS(block_size_t) )offsetof( NS(Particles), p0c ),
            ( NS(block_size_t) )offsetof( NS(Particles), s ),
            ( NS(block_size_t) )offsetof( NS(Particles), x ),
            ( NS(block_size_t) )offsetof( NS(Particles), y ),
            ( NS(block_size_t) )offsetof( NS(Particles), px ),
            ( NS(block_size_t) )offsetof( NS(Particles), py ),
            ( NS(block_size_t) )offsetof( NS(Particles), sigma ),
            ( NS(block_size_t) )offsetof( NS(Particles), psigma ),
            ( NS(block_size_t) )offsetof( NS(Particles), delta ),
            ( NS(block_size_t) )offsetof( NS(Particles), rpp ),
            ( NS(block_size_t) )offsetof( NS(Particles), rvv ),
            ( NS(block_size_t) )offsetof( NS(Particles), chi ),
            ( NS(block_size_t) )offsetof( NS(Particles), particle_id ),
            ( NS(block_size_t) )offsetof( NS(Particles), lost_at_element_id ),
            ( NS(block_size_t) )offsetof( NS(Particles), lost_at_turn ),
            ( NS(block_size_t) )offsetof( NS(Particles), state )        
        };
        
        NS(block_size_t) const data_attr_counts[] =
        {
            num, num, num, num, num, num, num, num, num, num,
            num, num, num, num, num, num, num, num, num, num 
        };
        
        NS(Particles_preset)( &particles );
        NS(Particles_set_num_particles)( &particles, num_of_particles );
        
        NS(BlockInfo)* ptr_info_block = NS(Blocks_add_block)(
            blocks, NS(BLOCK_TYPE_PARTICLE), sizeof( particles ), &particles,
            NUM_ATTR_DATA_POINTERS, data_attr_offsets, data_attr_sizes, 
                data_attr_counts );
        
        if( ptr_info_block != 0 )
        {
            ptr_particles = ( SIXTRL_GLOBAL_DEC NS(Particles)*                     
                )NS(BlockInfo_get_const_ptr_begin)( ptr_info_block );
        }
    }
    
    return ptr_particles;
}

/* end: sixtracklib/common/details/particles.c */
