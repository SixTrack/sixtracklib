#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles,                                                            
    NS(block_num_elements_t) const ii, 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_exact_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, 
    const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_multipole_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, 
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_TRACK_RETURN 
NS(Track_range_of_particles_over_beam_element)(    
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) index,
    NS(block_num_elements_t) const index_end,    
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT be_block_it );

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( _GPUCODE )

#include "sixtracklib/common/impl/beam_elements_api.h"
#include "sixtracklib/common/impl/particles_api.h"

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_drift_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    
    SIXTRL_REAL_T const length = NS(Drift_get_length)( drift );
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_drift_exact_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii,
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_STATIC SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1u;
    
    SIXTRL_REAL_T const length = NS(DriftExact_get_length)( drift );
    SIXTRL_REAL_T const delta  = NS(Particles_get_delta_value)( particles, ii );
    SIXTRL_REAL_T const beta0  = NS(Particles_get_beta0_value)( particles, ii );
    SIXTRL_REAL_T const px     = NS(Particles_get_px_value)(    particles, ii );
    SIXTRL_REAL_T const py     = NS(Particles_get_py_value)(    particles, ii );
    SIXTRL_REAL_T sigma        = NS(Particles_get_sigma_value)( particles, ii );
                        
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_multipole_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, 
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_INT64_T const order = NS(MultiPole_get_order)( multipole );
    SIXTRL_INT64_T index_x = 2 * order;
    SIXTRL_INT64_T index_y = index_x + 1;
        
    SIXTRL_REAL_T dpx = NS(MultiPole_get_bal_value)( multipole, index_x );    
    SIXTRL_REAL_T dpy = NS(MultiPole_get_bal_value)( multipole, index_y );
        
    SIXTRL_REAL_T const x   = NS(Particles_get_x_value)( particles, ii );
    SIXTRL_REAL_T const y   = NS(Particles_get_y_value)( particles, ii );
    SIXTRL_REAL_T const chi = NS(Particles_get_chi_value)( particles, ii );
    SIXTRL_REAL_T const len = NS(MultiPole_get_length)( multipole );
    
    SIXTRL_REAL_T px = NS(Particles_get_px_value)( particles, ii );
    SIXTRL_REAL_T py = NS(Particles_get_py_value)( particles, ii );
    
    for( ; index_x >= 0 ; index_x -= 2, index_y -= 2 )
    {
        SIXTRL_REAL_T const zre = dpx * x - dpy * y;
        SIXTRL_REAL_T const zim = dpx * y + dpy * x;
        
        dpx = NS(MultiPole_get_bal_value)( multipole, index_x ) + zre;
        dpy = NS(MultiPole_get_bal_value)( multipole, index_y ) + zim;
    }
    
    dpx = -chi * dpx;
    dpy =  chi * dpy;
    
    if( len > 0.0 )
    {
        SIXTRL_REAL_T const hxl   = NS(MultiPole_get_hxl)( multipole );
        SIXTRL_REAL_T const hyl   = NS(MultiPole_get_hyl)( multipole );            
        
        SIXTRL_REAL_T const delta = 
            NS(Particles_get_delta_value)( particles, ii );
            
        SIXTRL_REAL_T const hxx   = x * hxl / len;
        SIXTRL_REAL_T const hyy   = y * hyl / len;
        
        SIXTRL_REAL_T sigma = NS(Particles_get_sigma_value)( particles, ii );
        
        dpx += hxl + hxl * delta 
             - hxx * chi * NS(MultiPole_get_bal_value)( multipole, 0 );
             
        dpy -= hyl + hyl * delta
             - chi * NS(MultiPole_get_bal_value)( multipole, 1 );  
             
        sigma -= chi * ( hxx - hyy ) * len 
               * NS(Particles_get_rvv_value)( particles, ii );
               
        NS(Particles_set_sigma_value)( particles, ii, sigma );
    }
    
    px += dpx;
    py += dpy;
    
    NS(Particles_set_px_value)( particles, ii, px );
    NS(Particles_set_py_value)( particles, ii, py );
    
    return 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN 
NS(Track_range_of_particles_over_beam_element)(    
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) index,
    NS(block_num_elements_t) const index_end,    
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT be_block_it )
{
    int ret = 0;
    
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( be_block_it );
    
    SIXTRL_GLOBAL_DEC void const* pr_beam_element_begin =
        NS(BlockInfo_get_const_ptr_begin)( be_block_it );
        
    #else /* !defined( _GPUCODE ) */
    
    NS(BlockInfo) const info    = *be_block_it;
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( &info );
    
    SIXTRL_GLOBAL_DEC void const* pr_beam_element_begin =
            NS(BlockInfo_get_const_ptr_begin)( &info );
    
    #endif /* !defined( _GPUCODE ) */
    
    SIXTRL_ASSERT( ptr_beam_element_begin != 0 );
            
    switch( type_id )
    {
        case NS(BLOCK_TYPE_DRIFT):
        {
            typedef NS(Drift) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_drift_particle)( particles, index, &be );
            }
            
            break;
        }
        
        case NS(BLOCK_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );            
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_drift_exact_particle)( particles, index, &be );
            }
            
            break;
        }
        
        case NS(BLOCK_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_multipole_particle)( particles, index, &be );
            }
            
            break;
        }
        
        default:
        {
            ret = -1;
        }
    };
    
    return ret;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#include "sixtracklib/common/impl/track_api.h"

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
