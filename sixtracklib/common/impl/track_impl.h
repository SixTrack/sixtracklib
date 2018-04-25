#ifndef SIXTRACKLIB_COMMON_IMPL_TRACK_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_TRACK_IMPL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/impl/block_drift_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
SIXTRL_STATIC int NS(Track_drift)( 
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, 
    SIXTRL_REAL_T const length );

SIXTRL_STATIC int NS(Track_drift_exact)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length );

/* ========================================================================= */
/* =====                                                               ===== */
/* =====            Implementation of inline functions                 ===== */
/* =====                                                               ===== */
/* ========================================================================= */

SIXTRL_INLINE int NS(Track_drift)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_SIZE_T const ip, SIXTRL_REAL_T const length )
{
    double const _rpp = NS(Particles_get_rpp_value)( particles, ip );
    double const _px  = NS(Particles_get_px_value )( particles, ip ) * _rpp; 
    double const _py  = NS(Particles_get_py_value )( particles, ip ) * _rpp;    
    double const dsigma = ( 1.0 - NS(Particles_get_rvv_value)( particles, ip ) * 
                          ( 1.0 + ( _px * _px + _py * _py ) / 2.0 ) );
    
    double const _sigma = NS(Particles_get_sigma_value)( particles, ip ) + length * dsigma;
    double const _s     = NS(Particles_get_s_value)( particles, ip ) + length;
    double const _x     = NS(Particles_get_x_value)( particles, ip ) + length * _px;
    double const _y     = NS(Particles_get_y_value)( particles, ip ) + length * _py;
    
    NS(Particles_set_s_value)( particles, ip, _s );
    NS(Particles_set_x_value)( particles, ip, _x );
    NS(Particles_set_y_value)( particles, ip, _y );
    NS(Particles_set_sigma_value)( particles, ip, _sigma );
        
    return 1;    
}

/* ------------------------------------------------------------------------- */

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

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )
    #ifdef __cplusplus
        }
    #endif /* __cplusplus */
#endif /* !defiend( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_TRACK_IMPL_H__ */

/* end: sixtracklib/common/impl/track_impl.h */
