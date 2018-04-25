#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/impl/block_drift_type.h"
#include "sixtracklib/common/impl/track_impl.h"
#include "sixtracklib/common/particles_sequence.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
/* ------------------------------------------------------------------------- */

SIXTRL_STATIC void NS(Track_single_particle_over_beam_element)(
    const NS(BeamElementInfo) *const SIXTRL_RESTRICT element,
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const particle_index );

SIXTRL_STATIC void NS(Track_single_particle)(
    NS(BeamElementInfo) const* SIXTRL_RESTRICT elements_it,
    SIXTRL_SIZE_T const num_of_elements,
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const particle_index, 
    NS(ParticlesSequence)* SIXTRL_RESTRICT elem_by_elem );

/* ========================================================================= */
/* =====                                                               ===== */
/* =====            Implementation of inline functions                 ===== */
/* =====                                                               ===== */
/* ========================================================================= */

SIXTRL_INLINE void NS(Track_single_particle_over_beam_element)(
    const NS(BeamElementInfo) *const SIXTRL_RESTRICT element,
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const particle_index )
{
    typedef NS(BeamElementType) type_id_t;
    typedef SIXTRL_INT64_T      element_id_t;
    
    typedef NS(Drift)       drift_t;
    typedef unsigned char*  ptr_t;
    
    SIXTRL_ASSERT( NS(BeamElementInfo_is_available)( element ) );
    SIXTRL_ASSERT( ( particles != 0 ) && 
                   ( NS(Particles_get_size)( particles ) > particle_index ) );
    
    type_id_t const type_id = NS(BeamElementInfo_get_type_id)( element );
    
    switch( type_id )
    {
        case NS(ELEMENT_TYPE_DRIFT):
        {
            drift_t drift;
            double length;
            
            #if !defined( NDEBUG )
            ptr_t next_ptr = NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            SIXTRL_ASSERT( next_ptr != 0 );
            #else 
            NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            #endif /* !defined( NDEBUG ) */
            
            length = NS(Drift_get_length)( &drift );
            NS(Track_drift)( particles, particle_index, length );
            
            break;
        }
        
        case NS(ELEMENT_TYPE_DRIFT_EXACT):
        {
            drift_t drift;
            #if !defined( NDEBUG )
            ptr_t next_ptr = NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            SIXTRL_ASSERT( next_ptr != 0 );
            #else 
            NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            #endif /* !defined( NDEBUG ) */
            
            NS(Track_drift_exact)( particles, particle_index, 
                             NS(Drift_get_length)( &drift ) );
            
            break;            
        }
        
        default:
        {
            element_id_t const beam_elem_id = 
                NS(BeamElementInfo_get_element_id)( element );
            
            NS(Particles_set_lost_at_element_id_value)( 
                particles, particle_index, beam_elem_id );
            
            NS(Particles_set_state_value)( particles, particle_index, -1 );
        }        
    };
    
    return;
}

SIXTRL_INLINE void NS(Track_single_particle)(
    NS(BeamElementInfo) const* SIXTRL_RESTRICT elements_it,
    SIXTRL_SIZE_T const num_of_elements,
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const particle_index, 
    NS(ParticlesSequence)* SIXTRL_RESTRICT elem_by_elem )
{
    SIXTRL_SIZE_T ii = 0;    
    
    SIXTRL_ASSERT( elements_it != 0 );
    
    if( elem_by_elem == 0 )
    {
        for( ; ii < num_of_elements ; ++ii )
        {
            NS(Track_single_particle_over_beam_element)( 
                &elements_it[ ii ], particles, particle_index );
        }
    }
    else
    {
        for( ; ii < num_of_elements ; ++ii )
        {
            NS(Track_single_particle_over_beam_element)( 
                &elements_it[ ii ], particles, particle_index );
            
            NS(Particles_copy_single_unchecked)(
                NS(ParticlesSequence_get_particles_by_index)(
                    elem_by_elem, ii ), particle_index, 
                particles, particle_index );
        }
    }
    
    return;
}

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
