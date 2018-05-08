#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Track_drift)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

SIXTRL_STATIC int NS(Track_drift_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

SIXTRL_STATIC int NS(Track_drift_exact_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, SIXTRL_REAL_T const length );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Track_beam_elements)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */


SIXTRL_INLINE int NS(Track_beam_elements)(NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    int status = 0;
    
    int const use_elem_by_elem_store = (
        ( elem_by_elem_buffer != 0 ) && 
        ( NS(ParticlesContainer_get_num_of_blocks)( elem_by_elem_buffer ) >=
          ( NS(BeamElements_get_num_of_blocks)( beam_elements ) + 
                elem_by_elem_start_index ) ) );
    
    NS(BlockInfo) const* be_block_info_it  = 
        NS(BeamElements_get_const_block_infos_begin)( beam_elements );
        
    NS(BlockInfo) const* be_block_info_end =
        NS(BeamElements_get_const_block_infos_end)( beam_elements );
    
    NS(block_num_elements_t) elem_by_elem_idx = elem_by_elem_start_index;
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_info_it != 0 ) );
        
    for( ; be_block_info_it != be_block_info_end ; ++be_block_info_it )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
        
        NS(BlockType) const type_id = 
            NS(BlockInfo_get_type_id)( be_block_info_it );
        
        int status = 0;    
            
        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift ); /* only to get rid of warnings! */
                
                status |= NS(Drift_remap_from_memory)( &drift, be_block_info_it, 
                    ( g_ptr_uchar_t )NS(BeamElements_get_const_ptr_data_begin)( 
                        beam_elements ), 
                    NS(BeamElements_get_data_capacity)( beam_elements ) );
                
                status |= NS(Track_drift)( 
                    particles, start_particle_index, end_particle_index, 
                        NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift );
                
                status |= NS(Drift_remap_from_memory)( &drift, be_block_info_it, 
                    ( g_ptr_uchar_t )NS(BeamElements_get_const_ptr_data_begin)( 
                        beam_elements ),
                    NS(BeamElements_get_data_capacity)( beam_elements ) );
                
                status |= NS(Track_drift_exact)( 
                    particles, start_particle_index, end_particle_index, 
                    NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            default:
            {
                status = -1;
            }
        };
        
        if( use_elem_by_elem_store )
        {
            NS(Particles) particles_after_elem;
             
            status |= NS(ParticlesContainer_get_particles)( 
                &particles_after_elem, elem_by_elem_buffer, elem_by_elem_idx );
            
            NS(Particles_copy_all_unchecked)( &particles_after_elem, particles );
            ++elem_by_elem_idx;
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_drift_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const length )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    
    SIXTRL_REAL_T const rpp = NS(Particles_get_rpp_value)( particles, ii );
    SIXTRL_REAL_T const px  = NS(Particles_get_px_value )( particles, ii ) * rpp;
    SIXTRL_REAL_T const py  = NS(Particles_get_py_value )( particles, ii ) * rpp;    
    
    SIXTRL_REAL_T const dsigma = 
        ( ONE - NS(Particles_get_rvv_value)( particles, ii ) * 
            ( ONE + ONE_HALF * ( px * px + py * py ) ) );
    
    SIXTRL_REAL_T sigma = NS(Particles_get_sigma_value)( particles, ii );
    SIXTRL_REAL_T s     = NS(Particles_get_s_value)( particles, ii );
    SIXTRL_REAL_T x     = NS(Particles_get_x_value)( particles, ii );
    SIXTRL_REAL_T y     = NS(Particles_get_y_value)( particles, ii );
    
    sigma += length * dsigma;
    s     += length;
    x     += length * px;
    y     += length * py;
    
    NS(Particles_set_s_value)( particles, ii, s );
    NS(Particles_set_x_value)( particles, ii, x );
    NS(Particles_set_y_value)( particles, ii, y );
    NS(Particles_set_sigma_value)( particles, ii, sigma );
    
    return 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Track_drift_exact_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const length )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1u;
    
    SIXTRL_REAL_T const delta = NS(Particles_get_delta_value)( particles, ii );
    SIXTRL_REAL_T const beta0 = NS(Particles_get_beta0_value)( particles, ii );
    SIXTRL_REAL_T const px    = NS(Particles_get_px_value)(    particles, ii );
    SIXTRL_REAL_T const py    = NS(Particles_get_py_value)(    particles, ii );
    SIXTRL_REAL_T sigma       = NS(Particles_get_sigma_value)( particles, ii );
                        
    SIXTRL_REAL_T const opd   = delta + ONE;
    SIXTRL_REAL_T const lpzi  = ( length ) / 
        sqrt( opd * opd - px * px - py * py );
    
    SIXTRL_REAL_T const lbzi  = ( beta0 * beta0 * sigma + ONE ) * lpzi;
    
    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, ii );
    SIXTRL_REAL_T y = NS(Particles_get_y_value)( particles, ii );
    SIXTRL_REAL_T s = NS(Particles_get_s_value)( particles, ii );
    
    x     += px * lpzi;
    y     += py * lpzi;
    s     += length;
    sigma += length - lbzi;
    
    NS(Particles_set_x_value)(     particles, ii, x     );
    NS(Particles_set_y_value)(     particles, ii, y     );
    NS(Particles_set_s_value)(     particles, ii, s     );
    NS(Particles_set_sigma_value)( particles, ii, sigma );
    
    return 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Track_drift)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length )
{
    int status = 0;
    
    NS(block_num_elements_t) ii = start_particle_index;
    
    SIXTRL_ASSERT( 
        ( start_particle_index <= end_particle_index ) &&
        ( end_particle_index <= NS(Particles_get_num_particles)( particles ) ) 
    );
    
    for( ; ii < end_particle_index ; ++ii )
    {
        status |= NS(Track_drift_particle)( particles, ii, length );
    }
    
    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length )
{
    int status = 0;
    
    NS(block_num_elements_t) ii = start_particle_index;
    
    SIXTRL_ASSERT( 
        ( start_particle_index <= end_particle_index ) &&
        ( end_particle_index <= NS(Particles_get_num_particles)( particles ) ) 
    );
    
    for( ; ii < end_particle_index ; ++ii )
    {
        status |= NS(Track_drift_exact_particle)( particles, ii, length );
    }
    
    return status;
}



#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
