#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__

#if !defined( _GPUCODE )
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/impl/block_drift_type.h"
#include "sixtracklib/common/single_particle.h"

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS( SingleParticle );

int NS( TrackSingle_drift )( 
    NS( SingleParticle )* SIXTRL_RESTRICT particle, SIXTRL_REAL_T const len );

int NS( TrackSingle_drift_exact )( 
    NS( SingleParticle )* SIXTRL_RESTRICT particle, SIXTRL_REAL_T const len );

SIXTRL_STATIC void NS(TrackSingle_single_particle_over_beam_element)(
    NS(BeamElementInfo) const* element, NS(SingleParticle)* particle );

SIXTRL_STATIC void NS(TrackSingle_single_particle)(
    NS(BeamElementInfo) const* begin, SIXTRL_SIZE_T const num_of_elements,
    NS(SingleParticle)* particle );



SIXTRL_INLINE void NS(TrackSingle_single_particle_over_beam_element)(
    const NS(BeamElementInfo) *const element, NS(SingleParticle)* particle )
{
    typedef NS(BeamElementType) type_id_t;
    typedef SIXTRL_INT64_T      element_id_t;
    
    typedef NS(DriftSingle) drift_t;
        
    SIXTRL_ASSERT( NS(BeamElementInfo_is_available)( element ) );
    SIXTRL_ASSERT( particle != 0 );
    
    type_id_t const type_id = NS(BeamElementInfo_get_type_id)( element );
    
    switch( type_id )
    {
        case NS(ELEMENT_TYPE_DRIFT):
        {
            drift_t* ptr_drift = ( drift_t* )( 
                NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            
            SIXTRL_ASSERT( ptr_drift != 0 );
            NS(TrackSingle_drift)( particle, ptr_drift->length );
            break;
        }
        
        case NS(ELEMENT_TYPE_DRIFT_EXACT):
        {
            drift_t* ptr_drift = ( drift_t* )( 
                NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            
            SIXTRL_ASSERT( ptr_drift != 0 );
            NS(TrackSingle_drift_exact)( particle, ptr_drift->length );
            break;
        }
        
        default:
        {
            element_id_t const beam_elem_id = 
                NS(BeamElementInfo_get_element_id)( element );
            
            particle->elemid = beam_elem_id;
            particle->state  = -1;
        }        
    };
    
    return;
}

SIXTRL_INLINE void NS(TrackSingle_single_particle)(
    NS(BeamElementInfo) const* elements, SIXTRL_SIZE_T const num_of_elements,
    NS(SingleParticle)* particle )
{
    SIXTRL_ASSERT( elements != 0 );
    
    NS(BeamElementInfo) const* it  = elements;
    NS(BeamElementInfo) const* end = elements + num_of_elements;
    
    for( ; it != end ; ++it )
    {
        NS(TrackSingle_single_particle_over_beam_element)( it, particle );
    }
    
    return;
}

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_SINGLE_H__ */
