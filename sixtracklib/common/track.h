#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#include "sixtracklib/_impl/definitions.h"

#if !defined( _GPUCODE )

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/particles_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
struct NS(Particles);

SIXTRL_STATIC int NS(Drift_track)( 
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length );

SIXTRL_STATIC int NS(DriftExact_track)(
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length );

/* -------------------------------------------------------------------------- */
/* ----                                                                  ---- */ 
/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_track)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_UINT64_T const ip, SIXTRL_REAL_T const length )
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


SIXTRL_INLINE int NS(DriftExact_track)(
    NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_UINT64_T const ip, 
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


SIXTRL_GPUKERNEL void NS(Block_track)(
    SIXTRL_SIZE_T const num_of_turns,
    SIXTRL_SIZE_T const num_of_elements, 
    SIXTRL_GLOBAL_DEC unsigned char* elements, 
    SIXTRL_GLOBAL_DEC unsigned char* particles, 
    SIXTRL_GLOBAL_DEC NS(Particles)** elem_by_eleme_ptr, 
    SIXTRL_GLOBAL_DEC NS(Particles)** turn_by_turn_ptr )
{
    bool const elem_by_elem_flag = ( elem_by_elem_ptr != 0 );
    bool const turn_by_turn_flag = ( turn_by_turn_ptr != 0 );
    
    SIXTRL_SIZE_T const 
    
}

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
