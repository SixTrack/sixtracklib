#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/impl/beam_elements_api.h"

/* ------------------------------------------------------------------------- */

extern SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_reserve_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

extern SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_reserve_drift_exact)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

extern SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_reserve_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_INT64_T const order );

extern SIXTRL_GLOBAL_DEC NS(BeamBeam)* NS(Blocks_reserve_beam_beam)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_slices );

/* ------------------------------------------------------------------------- */

SIXTRL_GLOBAL_DEC NS(Drift)* NS(Blocks_reserve_drift)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    SIXTRL_GLOBAL_DEC NS(Drift)* ptr_drift = 0;
    
    if( blocks != 0 )
    {
        NS(Drift) drift;
        NS(Drift_preset)( &drift );
        
        NS(BlockInfo)* ptr_info_block = NS(Blocks_add_block)( blocks, 
            NS(BLOCK_TYPE_DRIFT), sizeof( drift ), &drift, 0u, 0, 0, 0 );
        
        if( ptr_info_block != 0 )
        {
            ptr_drift = NS(Blocks_get_drift)( ptr_info_block );
        }
    }
    
    return ptr_drift;
}

/* ------------------------------------------------------------------------- */

SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(Blocks_reserve_drift_exact)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    SIXTRL_GLOBAL_DEC NS(DriftExact)* ptr_drift = 0;
    
    if( blocks != 0 )
    {
        NS(DriftExact) drift;
        NS(DriftExact_preset)( &drift );
        
        NS(BlockInfo)* ptr_info_block = NS(Blocks_add_block)( blocks, 
            NS(BLOCK_TYPE_DRIFT_EXACT), sizeof( drift ), &drift, 0u, 0, 0, 0 );
        
        if( ptr_info_block != 0 )
        {
            ptr_drift = NS(Blocks_get_drift_exact)( ptr_info_block );
        }
    }
    
    return ptr_drift;
}

/* ------------------------------------------------------------------------- */

SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(Blocks_reserve_multipole)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, SIXTRL_INT64_T const ord )
{
    SIXTRL_GLOBAL_DEC NS(MultiPole)* ptr_multi_pole = 0;
    
    SIXTRL_STATIC NS(block_size_t) const NUM_ATTR_DATA_POINTERS = 1u;
    SIXTRL_STATIC NS(block_size_t) const REAL_SIZE = sizeof( SIXTRL_REAL_T  );
    
    NS(block_size_t) const data_attr_sizes[] = { REAL_SIZE };
    
    if( ( blocks != 0 ) && ( ord > 0 ) )
    {
        NS(MultiPole) multipole;
        NS(MultiPole_preset)( &multipole );
        NS(MultiPole_set_order)( &multipole, ord );
        
        NS(block_size_t) const num_bal_values = 
            2u * ( ( ( NS(block_size_t) )ord ) + 1u );
        
        NS(block_size_t) const data_attr_counts[] = { num_bal_values };
        
        NS(block_size_t) const data_attr_offsets[] = 
        {
            ( NS(block_size_t) )offsetof( NS(MultiPole), bal )
        };
                
        NS(BlockInfo)* ptr_info_block = NS(Blocks_add_block)( blocks,
            NS(BLOCK_TYPE_MULTIPOLE), sizeof( multipole ), &multipole,
            NUM_ATTR_DATA_POINTERS, data_attr_offsets, data_attr_sizes, 
                data_attr_counts );
        
        if( ptr_info_block != 0 )
        {
            ptr_multi_pole = NS(Blocks_get_multipole)( ptr_info_block );
        }
    }
    
    return ptr_multi_pole;
}

/* ------------------------------------------------------------------------- */

SIXTRL_GLOBAL_DEC NS(BeamBeam)* NS(Blocks_reserve_beam_beam)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_slices )
{
    SIXTRL_GLOBAL_DEC NS(BeamBeam)* ptr_beam_beam = 0;
    
    SIXTRL_STATIC NS(block_size_t) const NUM_ATTR_DATA_POINTERS = 6u;
    
    if( ( blocks != 0 ) && 
        ( num_of_slices > ( NS(block_num_elements_t) )0u ) )
    {
        SIXTRL_STATIC_VAR NS(block_size_t) const ONE = ( NS(block_size_t) )1u;            
        
        NS(BeamBeam) beam_beam;
        NS(BeamBeam_preset)( &beam_beam );
        NS(BeamBeam_set_num_of_slices)( &beam_beam, num_of_slices );
        
        NS(block_size_t) const N = ( NS(block_size_t) )num_of_slices;
        
        NS(block_size_t) const data_attr_counts[] = 
        {
            ONE, ONE, N, N, N, N             
        };
        
        NS(block_size_t) const data_attr_sizes[] = 
        {
            sizeof( NS(BeamBeamBoostData) ), 
            sizeof( NS(BeamBeamSigmas) ), 
            sizeof( SIXTRL_REAL_T ), 
            sizeof( SIXTRL_REAL_T ),
            sizeof( SIXTRL_REAL_T ), 
            sizeof( SIXTRL_REAL_T ) 
        };
        
        NS(block_size_t) const data_attr_offsets[] = 
        {
            ( NS(block_size_t) )offsetof( NS(BeamBeam), boost  ),
            ( NS(block_size_t) )offsetof( NS(BeamBeam), sigmas ),
            ( NS(block_size_t) )offsetof( NS(BeamBeam), n_part_per_slice ),
            ( NS(block_size_t) )offsetof( NS(BeamBeam), x_slices_star ),
            ( NS(block_size_t) )offsetof( NS(BeamBeam), y_slices_star ),
            ( NS(block_size_t) )offsetof( NS(BeamBeam), sigma_slices_star )
        };
                
        NS(BlockInfo)* ptr_info_block = NS(Blocks_add_block)( blocks,
            NS(BLOCK_TYPE_BEAM_BEAM), sizeof( beam_beam ), &beam_beam,
            NUM_ATTR_DATA_POINTERS, data_attr_offsets, data_attr_sizes, 
                data_attr_counts );
        
        if( ptr_info_block != 0 )
        {
            ptr_beam_beam = NS(Blocks_get_beam_beam)( ptr_info_block );
        }
    }
    
    return ptr_beam_beam;
}

/* end: sixtracklib/common/details/beam_elements.c */
