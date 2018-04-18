#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#include "sixtracklib/_impl/definitions.h"

#if !defined( _GPUCODE )

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/impl/block_drift_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS(Particles);
struct NS(ParticlesSequence);
struct NS(BeamElementInfo);

SIXTRL_STATIC int NS(Track_drift)( 
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length );

SIXTRL_STATIC int NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length );

/* -------------------------------------------------------------------------- */
/* ----                                                                  ---- */ 
/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_drift)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const ip, SIXTRL_REAL_T const length )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC SIXTRL_REAL_T const TWO = ( SIXTRL_REAL_T )2;
    
    SIXTRL_REAL_T const rpp = NS(Particles_get_rpp_value)( particles,ip );
    SIXTRL_REAL_T const rvv = NS(Particles_get_rvv_value)( particles,ip );
    SIXTRL_REAL_T const px = NS(Particles_get_px_value)(particles, ip ) * rpp;
    SIXTRL_REAL_T const py = NS(Particles_get_py_value)( particles, ip ) * rpp;
    SIXTRL_REAL_T const dsigma = 
        ( ONE - rvv * ( ONE + ( px * px + py * py ) / TWO ) );
        
    SIXTRL_REAL_T x     = NS(Particles_get_x_value)( particles, ip );
    SIXTRL_REAL_T y     = NS(Particles_get_y_value)( particles, ip );
    SIXTRL_REAL_T s     = NS(Particles_get_s_value)( particles, ip );
    SIXTRL_REAL_T sigma = NS(Particles_get_sigma_value)( particles, ip );
    
    x     += length * px;
    y     += length * py;
    sigma += length * dsigma;
    s     += length;
    
    NS(Particles_set_sigma_value)( particles, ip, sigma );
    NS(Particles_set_x_value)(     particles, ip, x     );
    NS(Particles_set_y_value)(     particles, ip, y     );
    NS(Particles_set_s_value)(     particles, ip, s     );        
    
    return 1;
}


SIXTRL_INLINE int NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_SIZE_T const ip, 
        SIXTRL_REAL_T const length )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1;
    SIXTRL_REAL_T const delta = NS(Particles_get_delta_value)( particles, ip );
    SIXTRL_REAL_T const beta0 = NS(Particles_get_beta0_value)( particles, ip );
    SIXTRL_REAL_T sigma       = NS(Particles_get_sigma_value)( particles, ip );
    SIXTRL_REAL_T const px    = NS(Particles_get_px_value)(    particles, ip );
    SIXTRL_REAL_T const py    = NS(Particles_get_py_value)(    particles, ip );
    
    SIXTRL_REAL_T const opd   = delta + ONE;
    SIXTRL_REAL_T const lpzi  = ( length ) / sqrt( opd * opd - px * px - py * py );
    SIXTRL_REAL_T const lbzi  = ( beta0 * beta0 * sigma + ONE ) * lpzi;
    
    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, ip );
    SIXTRL_REAL_T y = NS(Particles_get_y_value)( particles, ip );
    SIXTRL_REAL_T s = NS(Particles_get_s_value)( particles, ip );
    
    x     += px * lpzi;
    y     += py * lpzi;
    s     += length;
    sigma += length - lbzi;
    
    NS(Particles_set_x_value)(     particles, ip, x     );
    NS(Particles_set_y_value)(     particles, ip, y     );
    NS(Particles_set_s_value)(     particles, ip, s     );
    NS(Particles_set_sigma_value)( particles, ip, sigma );
    
    return 1;
}

void NS(Track_single_particle_over_beam_element)(
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
            
            #if defined( _NDEBUG )
            ptr_t next_ptr = NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            #else 
            NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            SIXTRL_ASSERT( next_ptr != 0 );
            #endif /* defined( _NDEBUG ) */
                                        
            NS(Track_drift)( particles, particle_index, 
                             NS(Drift_get_length)( &drift ) );
            
            break;
        }
        
        case NS(ELEMENT_TYPE_DRIFT_EXACT):
        {
            drift_t drift;
            #if defined( _NDEBUG )
            ptr_t next_ptr = NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            #else 
            NS(Drift_unpack_from_flat_memory)( &drift, 
                ( ptr_t )NS(BeamElementInfo_get_const_ptr_mem_begin)( element ) );
            SIXTRL_ASSERT( next_ptr != 0 );
            #endif /* defined( _NDEBUG ) */
            
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

void NS(Track_single_particle)(
    NS(BeamElementInfo) const* SIXTRL_RESTRICT elements_it,
    SIXTRL_SIZE_T const num_of_elements,
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const particle_index, 
    NS(ParticlesSequence)* SIXTRL_RESTRICT elem_by_elem )
{
    SIXTRL_ASSERT( elements_it != 0 );
    
    NS(BeamElementInfo) const* elements_end = elements_it + num_of_elements;
    
    if( elem_by_elem == 0 )
    {
        for( ; elements_it != elements_end ; ++elements_it )
        {
            NS(Track_single_particle_over_beam_element)( 
                elements_it, particles, particle_index );
        }
    }
    else
    {
        SIXTRL_SIZE_T element_index = ( SIXTRL_SIZE_T )0u;
        
        for( ; elements_it != elements_end ; ++elements_it, ++element_index )
        {
            NS(Track_single_particle_over_beam_element)( 
                elements_it, particles, particle_index );
            
            NS(Particles_copy_single_unchecked)(
                NS(ParticlesSequence_get_particles_by_index)(
                    elem_by_elem, element_index ), particle_index, 
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
