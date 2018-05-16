#include "sixtracklib/common/particles.h"

#if !defined( _GPUCODE )

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"

#include "sixtracklib/common/alignment.h"
#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/block_info.h"
#include "sixtracklib/common/blocks_container.h"


extern int NS(ParticlesContainer_add_blocks_of_particles)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(Particles)* SIXTRL_RESTRICT particle_blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const* SIXTRL_RESTRICT num_of_particles_vec );

extern int NS(Particles_write_to_bin_file)( 
    FILE* fp, const NS(Particles) *const SIXTRL_RESTRICT particles );

extern NS(block_num_elements_t) 
    NS(Particles_get_next_num_particles_from_bin_file)( FILE* fp );

extern int NS(Particles_read_from_bin_file)( 
    FILE* fp, NS(Particles)* SIXTRL_RESTRICT particles );

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

/* ------------------------------------------------------------------------- */

int NS(Particles_write_to_bin_file)( 
    FILE* fp, const NS(Particles) *const SIXTRL_RESTRICT p )
{
    int success = -1;
    
    NS(block_num_elements_t) const num_of_particles = 
        NS(Particles_get_num_particles)( p );
    
    if( ( fp != 0 ) && ( num_of_particles > ( NS(block_num_elements_t) )0u ) )
    {
        static size_t const ONE = ( size_t )1u;        
        size_t const  REAL_SIZE = sizeof( SIXTRL_REAL_T );
        size_t const  I64_SIZE  = sizeof( SIXTRL_INT64_T );
        
        NS(block_size_t) const num_attributes = ( NS(block_size_t) )20u;
        NS(block_num_elements_t) const num_elements = 1;
            
        NS(block_size_t) const attr_sizes[] =
        {
            I64_SIZE,   I64_SIZE,   REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  
            REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  
            REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  REAL_SIZE,
            REAL_SIZE,  REAL_SIZE,  REAL_SIZE,  I64_SIZE,   I64_SIZE 
        };
        
        NS(block_size_t) const attr_counts[] = 
        {
            ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, 
            ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE
        };
        
        void* attr[] =
        {
            ( void* )NS(Particles_get_const_particle_id)( p ),
            ( void* )NS(Particles_get_const_state)( p ),
            ( void* )NS(Particles_get_const_q0)( p ),
            ( void* )NS(Particles_get_const_mass0)( p ),             
            ( void* )NS(Particles_get_const_beta0)( p ),
            ( void* )NS(Particles_get_const_gamma0)( p ),
            ( void* )NS(Particles_get_const_p0c)( p ), 
            ( void* )NS(Particles_get_const_s)( p ),
            ( void* )NS(Particles_get_const_x)( p ),
            ( void* )NS(Particles_get_const_y)( p ),
            ( void* )NS(Particles_get_const_px)( p ),
            ( void* )NS(Particles_get_const_py)( p ),
            ( void* )NS(Particles_get_const_sigma)( p ),
            ( void* )NS(Particles_get_const_psigma)( p ),
            ( void* )NS(Particles_get_const_delta)( p ), 
            ( void* )NS(Particles_get_const_rpp)( p ),
            ( void* )NS(Particles_get_const_rvv)( p ),
            ( void* )NS(Particles_get_const_chi)( p ),
            ( void* )NS(Particles_get_const_lost_at_element_id)( p ),
            ( void* )NS(Particles_get_const_lost_at_turn)( p )
        };
        
        success = NS(Block_write_to_binary_file)( fp, NS(BLOCK_TYPE_PARTICLE), 
            num_elements, num_attributes, attr, attr_sizes, attr_counts );
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

NS(block_num_elements_t) 
NS(Particles_get_next_num_particles_from_bin_file)( FILE* fp )
{
    SIXTRL_STATIC NS(block_size_t) const ONE      = ( NS(block_size_t) )1u;
    SIXTRL_STATIC NS(block_size_t) const ZERO     = ( NS(block_size_t) )0u;
    SIXTRL_STATIC SIXTRL_UINT64_T  const U64_ZERO = ( SIXTRL_UINT64_T  )0u;
    
    NS(block_size_t) const NUM_ATTRIBUTES = ( NS(block_size_t) )20u;
    NS(block_size_t) attr_sizes[] =
    {
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
    };
    
    NS(block_size_t) attr_counts[] = 
    {
        ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, 
        ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE
    };
    
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);
    NS(block_num_elements_t) num_particles = 0;
    NS(block_size_t) num_attrs = ZERO;
    SIXTRL_UINT64_T binary_length = U64_ZERO;
    SIXTRL_INT64_T success_flag = 0;
    
    if( ( 0 == NS(Block_peak_at_next_block_in_binary_file)( fp, &binary_length, 
            &success_flag, &type_id, &num_particles, &num_attrs, 
                &attr_sizes[ 0 ], &attr_counts[ 0 ], NUM_ATTRIBUTES ) ) &&
        ( num_attrs == NUM_ATTRIBUTES ) && 
        ( type_id == NS(BLOCK_TYPE_PARTICLE) ) && 
        ( success_flag == 0 ) && ( binary_length > ZERO ) )
    {
        return num_particles;
    }
    
    return ZERO;
}


int NS(Particles_read_from_bin_file)( 
    FILE* fp, NS(Particles)* SIXTRL_RESTRICT p )
{
    int success = -1;
    
    SIXTRL_STATIC NS(block_size_t) const ONE      = ( NS(block_size_t) )1u;
    SIXTRL_STATIC NS(block_size_t) const ZERO     = ( NS(block_size_t) )0u;
    SIXTRL_STATIC SIXTRL_UINT64_T  const U64_ZERO = ( SIXTRL_UINT64_T  )0u;
    
    NS(block_size_t) const NUM_ATTRIBUTES = ( NS(block_size_t) )20u;    
    
    NS(block_size_t) attr_sizes[] =
    {
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
    };
    
    NS(block_size_t) attr_counts[] = 
    {
        ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, 
        ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE
    };
    
    void* attr[] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    
    NS(BlockType) type_id = NS(BLOCK_TYPE_INVALID);
    NS(block_num_elements_t) num_particles = 0;
    NS(block_size_t) num_attrs = ZERO;
    SIXTRL_UINT64_T binary_length = U64_ZERO;
    SIXTRL_INT64_T success_flag  = 0;
    
    if( ( 0 == NS(Block_peak_at_next_block_in_binary_file)( fp, &binary_length, 
            &success_flag, &type_id, &num_particles, &num_attrs, 
                &attr_sizes[ 0 ], &attr_counts[ 0 ], NUM_ATTRIBUTES ) ) &&
        ( num_attrs == NUM_ATTRIBUTES ) &&
        ( type_id == NS(BLOCK_TYPE_PARTICLE) ) && ( success_flag == 0 ) && 
        ( binary_length > ZERO ) && ( num_particles > ZERO ) &&
        ( num_particles <= NS(Particles_get_num_particles)( p ) ) )
    {
        NS(block_num_elements_t) num_elements = 0;
        
        attr[  0 ] = ( void* )NS(Particles_get_particle_id)( p );
        attr[  1 ] = ( void* )NS(Particles_get_state)( p );       
        attr[  2 ] = ( void* )NS(Particles_get_q0)( p );     
        attr[  3 ] = ( void* )NS(Particles_get_mass0)( p );                
        attr[  4 ] = ( void* )NS(Particles_get_beta0)( p );  
        attr[  5 ] = ( void* )NS(Particles_get_gamma0)( p );
        attr[  6 ] = ( void* )NS(Particles_get_p0c)( p );    
        attr[  7 ] = ( void* )NS(Particles_get_s)( p );            
        attr[  8 ] = ( void* )NS(Particles_get_x)( p );      
        attr[  9 ] = ( void* )NS(Particles_get_y)( p );
        attr[ 10 ] = ( void* )NS(Particles_get_px)( p );     
        attr[ 11 ] = ( void* )NS(Particles_get_py)( p );       
        attr[ 12 ] = ( void* )NS(Particles_get_sigma)( p );  
        attr[ 13 ] = ( void* )NS(Particles_get_psigma)( p );
        attr[ 14 ] = ( void* )NS(Particles_get_delta)( p );  
        attr[ 15 ] = ( void* )NS(Particles_get_rpp)( p );       
        attr[ 16 ] = ( void* )NS(Particles_get_rvv)( p );    
        attr[ 17 ] = ( void* )NS(Particles_get_chi)( p );
        attr[ 18 ] = ( void* )NS(Particles_get_lost_at_element_id)( p );
        attr[ 19 ] = ( void* )NS(Particles_get_lost_at_turn)( p );
        
        success = NS(Block_read_structure_from_binary_file)( 
            fp, &binary_length, &success_flag, &type_id, &num_elements,
            &num_attrs, attr, &attr_sizes[ 0 ], &attr_counts[ 0 ] );
        
        SIXTRL_ASSERT( 
            ( success == 0 ) ||
            ( ( success_flag == 0 ) && ( binary_length > ZERO ) &&
              ( attr_counts[  0 ] == ONE ) && ( attr_counts[  1 ] == ONE ) && 
              ( attr_counts[  2 ] == ONE ) && ( attr_counts[  3 ] == ONE ) && 
              ( attr_counts[  4 ] == ONE ) && ( attr_counts[  5 ] == ONE ) && 
              ( attr_counts[  6 ] == ONE ) && ( attr_counts[  7 ] == ONE ) && 
              ( attr_counts[  8 ] == ONE ) && ( attr_counts[  9 ] == ONE ) && 
              ( attr_counts[ 10 ] == ONE ) && ( attr_counts[ 11 ] == ONE ) && 
              ( attr_counts[ 12 ] == ONE ) && ( attr_counts[ 13 ] == ONE ) && 
              ( attr_counts[ 14 ] == ONE ) && ( attr_counts[ 15 ] == ONE ) && 
              ( attr_counts[ 16 ] == ONE ) && ( attr_counts[ 17 ] == ONE ) && 
              ( attr_counts[ 18 ] == ONE ) && ( attr_counts[ 19 ] == ONE ) &&
              ( attr_sizes[   0 ] == sizeof( SIXTRL_INT64_T ) ) && 
              ( attr_sizes[   1 ] == sizeof( SIXTRL_INT64_T ) ) &&
              ( attr_sizes[   2 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   3 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   4 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   5 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   6 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   7 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   8 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[   9 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  10 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  11 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  12 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  13 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  14 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  15 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  16 ] == sizeof( SIXTRL_REAL_T  ) ) &&
              ( attr_sizes[  17 ] == sizeof( SIXTRL_REAL_T  ) ) &&              
              ( attr_sizes[  18 ] == sizeof( SIXTRL_INT64_T ) ) &&
              ( attr_sizes[  19 ] == sizeof( SIXTRL_INT64_T ) ) ) );              
    }
    
    return success;
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/details/particles.c */
