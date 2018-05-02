#include "sixtracklib/common/be_drift.h"
#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"

/* -------------------------------------------------------------------------- */

extern int NS(Drift_create_from_single_drift)( 
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift );

extern void NS(DriftSingle_track_particle)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift  );

extern void NS(Drift_exact_track_particle)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift  );

extern int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift );

/* -------------------------------------------------------------------------- */

int NS(Drift_create_from_single_drift)( 
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift )
{
    int success = -1;
    
    if( ( drift != 0 ) && ( single_drift != 0 ) )
    {
        NS(Drift_set_type_id_num)( 
            drift, NS(DriftSingle_get_type_id_num)( single_drift ) );
        
        NS(Drift_set_num_elements)( drift, ( NS(block_num_elements_t) )1u );
        
        NS(Drift_assign_ptr_to_element_id)( 
            drift, NS(DriftSingle_get_element_id_ptr)( single_drift ) );
        
        NS(Drift_assign_ptr_to_length)( 
            drift, NS(DriftSingle_get_length_ptr)( single_drift ) );
        
        success = 0;
    }
    else if( drift != 0 )
    {
        NS(Drift_preset)( drift );
    }
    
    return success;
}

void NS(DriftSingle_track_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    NS(Drift) drift;
    
    if( ( NS(BlockType_to_number)( NS(BLOCK_TYPE_DRIFT) ) ==
          NS(DriftSingle_get_type_id_num)( single_drift ) ) &&
        ( NS(Drift_create_from_single_drift)( 
            &drift, ( NS(DriftSingle)* )single_drift ) ) ) 
    {
        NS(Drift_track_particle_over_single_elem)( 
            particles, particle_index, &drift, 0u );
    }
    
    return;
}

void NS(DriftSingleExact_track_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_UINT64_T const particle_index,
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    NS(Drift) drift;
    
    if( ( NS(BlockType_to_number)( NS(BLOCK_TYPE_DRIFT_EXACT) ) ==
          NS(DriftSingle_get_type_id_num)( single_drift ) ) &&
        ( NS(Drift_create_from_single_drift)( 
            &drift, ( NS(DriftSingle)* )single_drift ) ) ) 
    {
        NS(DriftExact_track_particle_over_single_elem)( 
            particles, particle_index, &drift, 0u );
    }
    
    return;
}

int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int is_valid = 0;
    
    SIXTRL_UINT64_T const num_elements = NS(Drift_get_num_elements)( drift );
    
    if( num_elements > 0u )
    {
        NS(BlockType) const type_id = NS(Drift_get_type_id)( drift );
        
        if( ( type_id == NS(BLOCK_TYPE_DRIFT )  ) ||
            ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) )
        {
            SIXTRL_UINT64_T ii = ( SIXTRL_UINT64_T )0u;
            
            is_valid = 1;
            
            for( ; ii < num_elements ; ++ii )
            {
                if( ( NS(Drift_get_element_id_value)( drift, ii ) < 0 ) ||
                    ( NS(Drift_get_length_value)( drift, ii ) < 
                        ( SIXTRL_REAL_T )0.0 ) )
                {
                    is_valid = 0;
                    break;
                }
            }
        }
    }
    
    return is_valid;
}


/* end: sixtracklib/common/details/be_drift.c */
