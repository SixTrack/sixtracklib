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
#include "sixtracklib/common/track.h"

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

int NS(Drift_is_valid)( const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int is_valid = 0;
    
    NS(BlockType) const type_id = NS(Drift_get_type_id)( drift );
        
    if( ( type_id == NS(BLOCK_TYPE_DRIFT )  ) ||
        ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT ) ) )
    {
        is_valid = ( ( NS(Drift_get_element_id_value)( drift ) >= 0 ) &&
            ( NS(Drift_get_length_value)( drift ) >= ( SIXTRL_REAL_T )0.0 ) );
    }
    
    return is_valid;
}


/* end: sixtracklib/common/details/be_drift.c */
