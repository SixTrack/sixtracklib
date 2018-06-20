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

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles,                                                            
    NS(block_num_elements_t) const ii, 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_drift_exact_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, 
    const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_multipole_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t)  const ii, 
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_align_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(Align) *const SIXTRL_RESTRICT align );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_cavity_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(Cavity) *const SIXTRL_RESTRICT cavity );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_beam_beam_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(BeamBeam_boost_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const index, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(BeamBeam_inv_boost_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const index, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
    NS(BeamBeam_propagate_sigma_matrix)(
        NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result, 
        NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ds_result, 
        const NS(BeamBeamSigmas) *const sigma_matrix, 
        SIXTRL_REAL_T  const s, 
        SIXTRL_REAL_T  const treshold_singular, 
        SIXTRL_INT64_T const handle_sigularity );
    
SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
    NS(BeamBeam_get_transverse_fields)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gx_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gy_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y, 
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const min_sigma_diff );
        
SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
    NS(BeamBeam_get_transverse_fields_gauss_round)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component, 
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma, 
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y );
        
SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
    NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y );
    
SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam_complex_error_function)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT result_real,
        SIXTRL_REAL_T* SIXTRL_RESTRICT result_imag,
        SIXTRL_REAL_T const input_real, SIXTRL_REAL_T const input_imag );
        
/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
NS(Track_range_of_particles_over_beam_element)(    
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) index,
    NS(block_num_elements_t) const index_end,    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* SIXTRL_RESTRICT be_block_it );

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
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    
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
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE = ( SIXTRL_REAL_T )1u;
    
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

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_align_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(Align) *const SIXTRL_RESTRICT align )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0u;
    
    SIXTRL_REAL_T const sz = NS(Align_get_sz)( align );
    SIXTRL_REAL_T const cz = NS(Align_get_cz)( align );
    
    SIXTRL_REAL_T x        = NS(Particles_get_x_value)(  particles, ii );
    SIXTRL_REAL_T y        = NS(Particles_get_y_value)(  particles, ii );
    SIXTRL_REAL_T px       = NS(Particles_get_px_value)( particles, ii );
    SIXTRL_REAL_T py       = NS(Particles_get_py_value)( particles, ii );
    
    SIXTRL_REAL_T temp     = cz * x - sz * y - NS(Align_get_dx)( align );
    
    y    =  sz * x + cz * y - NS(Align_get_dy)( align );    
    x    =  temp;
    
    temp =  cz * px + sz * py;
    py   = -sz * px + cz * py;
    px   =  temp;
    
    NS(Particles_set_x_value)(  particles, ii, x  );
    NS(Particles_set_y_value)(  particles, ii, y  );
    NS(Particles_set_px_value)( particles, ii, px );
    NS(Particles_set_py_value)( particles, ii, py );
    
    return ret;
}

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_cavity_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0u;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO  = ( SIXTRL_REAL_T )2.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const PI   = 
        ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const CLIGHT = ( SIXTRL_REAL_T )299792458u;
        
    SIXTRL_REAL_T const beta0 = NS(Particles_get_beta0_value)( particles, ii );
    
    SIXTRL_REAL_T const phase = NS(Cavity_get_lag)( cavity ) -  
        ( TWO * PI * NS(Cavity_get_frequency)( cavity ) * 
            ( NS(Particles_get_sigma_value)( particles, ii ) / beta0 ) 
        ) / CLIGHT;
        
    SIXTRL_REAL_T const psigma = 
        NS(Particles_get_psigma_value)( particles, ii ) +
        ( NS(Particles_get_chi_value)( particles, ii ) * 
            NS(Cavity_get_voltage)( cavity ) * sin( phase ) ) /
                ( NS(Particles_get_p0c_value)( particles, ii ) * beta0 );
    
    SIXTRL_REAL_T const pt    = psigma * beta0;
    SIXTRL_REAL_T const opd   = sqrt( pt * pt + TWO * psigma + ONE );
    SIXTRL_REAL_T const beta  = opd / ( ONE / beta0 + pt );
    
    NS(Particles_set_psigma_value)( particles, ii, psigma       );
    NS(Particles_set_delta_value)(  particles, ii, opd   - ONE  );
    NS(Particles_set_rpp_value)(    particles, ii, ONE   / opd  );
    NS(Particles_set_rvv_value)(    particles, ii, beta0 / beta );
    
    return ret;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(BeamBeam_boost_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const index, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;
    
    #if !defined( NDEBUG ) && !defined( __CUDACC__ )
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const MIN_EPS = ( SIXTRL_REAL_T )1e-12L; /* ?? */
    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0L;
    
    SIXTRL_REAL_T px      = NS(Particles_get_px_value)(    particles, index );
    SIXTRL_REAL_T py      = NS(Particles_get_px_value)(    particles, index );
    SIXTRL_REAL_T delta   = NS(Particles_get_delta_value)( particles, index );
    
    SIXTRL_REAL_T const tphi   = NS(BeamBeamBoostData_get_tphi)( boost );
    SIXTRL_REAL_T const salpha = NS(BeamBeamBoostData_get_salpha)( boost );
    SIXTRL_REAL_T const calpha = NS(BeamBeamBoostData_get_calpha)( boost );
    
    SIXTRL_REAL_T const SALPHA_TPHI = salpha * tphi;
    SIXTRL_REAL_T const CALPHA_TPHI = calpha * tphi;
    
    SIXTRL_REAL_T const cphi = NS(BeamBeamBoostData_get_cphi)( boost );    
    SIXTRL_REAL_T const delta_plus_one = delta + ONE;
    SIXTRL_REAL_T temp  = delta_plus_one * delta_plus_one - px * px - py * py;
    SIXTRL_REAL_T const h = delta_plus_one - sqrt( temp );
    
    SIXTRL_REAL_T const delta_st    = 
        delta - px * CALPHA_TPHI - py * SALPHA_TPHI + h * tphi * tphi;
    
    SIXTRL_REAL_T const delta_st_plus_one = delta_st + ONE;
    SIXTRL_REAL_T const px_st       = px / cphi - h * CALPHA_TPHI / cphi;
    SIXTRL_REAL_T const py_st       = py / cphi - h * SALPHA_TPHI / cphi;
    SIXTRL_REAL_T const pz_st       = 
        sqrt( delta_st_plus_one * delta_st_plus_one - 
            px_st * px_st - py_st * py_st );
    
    SIXTRL_REAL_T const hx_st       = px_st / pz_st;
    SIXTRL_REAL_T const hy_st       = py_st / pz_st;
    SIXTRL_REAL_T const hsigma_st   = ONE - delta_st_plus_one / pz_st;
    
    SIXTRL_REAL_T const sphi        = NS(BeamBeamBoostData_get_sphi)( boost );
    SIXTRL_REAL_T const SALPHA_SPHI = salpha * sphi;
    SIXTRL_REAL_T const CALPHA_SPHI = calpha * sphi;
    
    SIXTRL_REAL_T const L11         = ONE + hx_st * CALPHA_SPHI;
    SIXTRL_REAL_T const L12         = hx_st * SALPHA_SPHI;
    SIXTRL_REAL_T const L13         = CALPHA_TPHI;
                                    
    SIXTRL_REAL_T const L21         = hy_st * CALPHA_SPHI;
    SIXTRL_REAL_T const L22         = ONE + hy_st * SALPHA_SPHI;
    SIXTRL_REAL_T const L23         = SALPHA_TPHI;
                                    
    SIXTRL_REAL_T const L31         = hsigma_st * CALPHA_SPHI;
    SIXTRL_REAL_T const L32         = hsigma_st * SALPHA_SPHI;
    SIXTRL_REAL_T const L33         = ONE / cphi;
    
    SIXTRL_REAL_T const x     = NS(Particles_get_x_value)(     particles, index );
    SIXTRL_REAL_T const y     = NS(Particles_get_y_value)(     particles, index );
    SIXTRL_REAL_T const sigma = NS(Particles_get_sigma_value)( particles, index );
    
    SIXTRL_REAL_T const x_star     = L11 * x + L12 * y + L13 * sigma;
    SIXTRL_REAL_T const y_star     = L21 * x + L22 * y + L23 * sigma;
    SIXTRL_REAL_T const sigma_star = L31 * x + L32 * y + L33 * sigma;
    
    NS(Particles_set_x_value)(     particles, index, x_star );
    NS(Particles_set_y_value)(     particles, index, y_star );
    NS(Particles_set_sigma_value)( particles, index, sigma_star );    
    NS(Particles_set_px_value)(    particles, index, px_st );
    NS(Particles_set_py_value)(    particles, index, py_st );
    NS(Particles_set_delta_value)( particles, index, delta_st );
    
    SIXTRL_ASSERT( 
        ( ( cphi  > MIN_EPS ) || ( -cphi  < MIN_EPS ) ) && 
        ( ( pz_st > MIN_EPS ) || ( -pz_st > MIN_EPS ) ) &&
        ( delta_plus_one * delta_plus_one >= ( px * px + py * py ) ) &&
        ( delta_st_plus_one * delta_st_plus_one >= 
            ( px_st * px_st + py_st * py_st ) ) );
    
    return ret;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(BeamBeam_inv_boost_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const index, 
    const NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;
    
    #if !defined( NDEBUG ) && !defined( __CUDACC__ )
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const MIN_EPS = ( SIXTRL_REAL_T )1e-12L; /* ?? */
    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0L;
    
    SIXTRL_REAL_T px_st    = NS(Particles_get_px_value)(    particles, index );
    SIXTRL_REAL_T py_st    = NS(Particles_get_px_value)(    particles, index );
    SIXTRL_REAL_T delta_st = NS(Particles_get_delta_value)( particles, index );
    
    SIXTRL_REAL_T const delta_st_plus_one = delta_st + ONE;
    
    SIXTRL_REAL_T const pz_st = sqrt( 
        delta_st_plus_one * delta_st_plus_one - px_st * px_st - py_st * py_st );
    
    SIXTRL_REAL_T const hx_st       = px_st / pz_st;
    SIXTRL_REAL_T const hy_st       = py_st / pz_st;
    SIXTRL_REAL_T const hsigma_st   = ONE - delta_st_plus_one / pz_st;
    
    SIXTRL_REAL_T const cphi     = NS(BeamBeamBoostData_get_cphi)( boost );
    SIXTRL_REAL_T const CPHI_INV = ONE / cphi;
    
    SIXTRL_REAL_T const sphi     = NS(BeamBeamBoostData_get_sphi)( boost );
    SIXTRL_REAL_T const tphi     = NS(BeamBeamBoostData_get_tphi)(   boost );
    SIXTRL_REAL_T const salpha   = NS(BeamBeamBoostData_get_salpha)( boost );
    SIXTRL_REAL_T const calpha   = NS(BeamBeamBoostData_get_calpha)( boost );
    
    SIXTRL_REAL_T const DET_L  = CPHI_INV + 
        ( hx_st * calpha + hy_st * salpha - hsigma_st * sphi ) * tphi;
        
    SIXTRL_REAL_T const INV_DET_L   = ONE / DET_L;    
    
    SIXTRL_REAL_T const SALPHA_TPHI = salpha * tphi;
    SIXTRL_REAL_T const CALPHA_TPHI = calpha * tphi;
    
    SIXTRL_REAL_T const SALPHA_SPHI = salpha * sphi;
    SIXTRL_REAL_T const CALPHA_SPHI = calpha * sphi;
    
    SIXTRL_REAL_T const L11_inv = INV_DET_L * ( CPHI_INV + SALPHA_TPHI * (
        hy_st - hsigma_st * SALPHA_SPHI ) );
    
    SIXTRL_REAL_T const L12_inv = INV_DET_L * ( SALPHA_TPHI * ( 
        hsigma_st * CALPHA_SPHI - hx_st ) );
    
    SIXTRL_REAL_T const L13_inv = -INV_DET_L * tphi * ( calpha - 
        SALPHA_SPHI * ( hx_st * salpha + hy_st * calpha ) );
    
    SIXTRL_REAL_T const L21_inv = INV_DET_L * ( 
        CALPHA_TPHI * (hsigma_st * SALPHA_SPHI - hy_st ) );
    
    SIXTRL_REAL_T const L22_inv = INV_DET_L * (
        CPHI_INV + CALPHA_TPHI * ( hx_st - hsigma_st * CALPHA_SPHI ) );
            
    SIXTRL_REAL_T const L23_inv = -INV_DET_L * tphi * ( salpha - 
        CALPHA_SPHI * ( hy_st * calpha + hx_st * salpha ) );
    
    SIXTRL_REAL_T const L31_inv = -INV_DET_L * hsigma_st * CALPHA_SPHI;
    SIXTRL_REAL_T const L32_inv = -INV_DET_L * hsigma_st * SALPHA_SPHI;
    SIXTRL_REAL_T const L33_inv =  INV_DET_L * 
        ( ONE + hx_st * CALPHA_SPHI + hy_st * SALPHA_SPHI );

    
    SIXTRL_REAL_T const x_st     = NS(Particles_get_x_value)(     particles, index );
    SIXTRL_REAL_T const y_st     = NS(Particles_get_y_value)(     particles, index );
    SIXTRL_REAL_T const sigma_st = NS(Particles_get_sigma_value)( particles, index );
    
    SIXTRL_REAL_T const x_inv     = 
        L11_inv * x_st + L12_inv * y_st + L13_inv * sigma_st;
        
    SIXTRL_REAL_T const y_inv     = 
        L21_inv * x_st + L22_inv * y_st + L23_inv * sigma_st;
        
    SIXTRL_REAL_T const sigma_inv = 
        L31_inv * x_st + L32_inv * y_st + L33_inv * sigma_st;
    
    SIXTRL_REAL_T const h      = ( delta_st_plus_one - pz_st ) * cphi * cphi;
    
    SIXTRL_REAL_T const px_inv    = px_st * cphi + h * CALPHA_TPHI;
    SIXTRL_REAL_T const py_inv    = py_st * cphi + h * SALPHA_TPHI;
    SIXTRL_REAL_T const delta_inv = delta_st + 
        px_inv * CALPHA_TPHI + py_inv * SALPHA_TPHI - h * tphi * tphi;
        
    NS(Particles_set_x_value)(     particles, index, x_inv );
    NS(Particles_set_y_value)(     particles, index, y_inv );
    NS(Particles_set_sigma_value)( particles, index, sigma_inv );
    
    NS(Particles_set_px_value)(    particles, index, px_inv );
    NS(Particles_set_py_value)(    particles, index, py_inv );
    NS(Particles_set_delta_value)( particles, index, delta_inv );
    
    SIXTRL_ASSERT( 
        ( ( cphi  > MIN_EPS ) || ( -cphi  > MIN_EPS ) ) && 
        ( ( pz_st > MIN_EPS ) || ( -pz_st > MIN_EPS ) ) &&
        ( ( DET_L > MIN_EPS ) || ( -DET_L > MIN_EPS ) ) );
    
    return ret;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(BeamBeam_propagate_sigma_matrix)(
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ptr_result, 
    NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ptr_deriv_result, 
    const NS(BeamBeamSigmas) *const sigmas, SIXTRL_REAL_T  const s, 
    SIXTRL_REAL_T  const treshold_singular, 
    SIXTRL_INT64_T const handle_sigularity )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const FOUR     = ( SIXTRL_REAL_T )4.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const EIGHT    = ( SIXTRL_REAL_T )8.0L;
    
    NS(BeamBeamPropagatedSigmasResult) result;
    NS(BeamBeamPropagatedSigmasResult) deriv_result;
    
    SIXTRL_REAL_T const s_squ    = s * s;
    
    SIXTRL_REAL_T const sigma_11 = NS(BeamBeamSigmas_get_sigma11)( sigmas );
    SIXTRL_REAL_T const sigma_12 = NS(BeamBeamSigmas_get_sigma12)( sigmas );
    SIXTRL_REAL_T const sigma_13 = NS(BeamBeamSigmas_get_sigma13)( sigmas );
    SIXTRL_REAL_T const sigma_14 = NS(BeamBeamSigmas_get_sigma14)( sigmas );    
    SIXTRL_REAL_T const sigma_22 = NS(BeamBeamSigmas_get_sigma22)( sigmas );
    SIXTRL_REAL_T const sigma_23 = NS(BeamBeamSigmas_get_sigma23)( sigmas );
    SIXTRL_REAL_T const sigma_24 = NS(BeamBeamSigmas_get_sigma24)( sigmas );    
    SIXTRL_REAL_T const sigma_33 = NS(BeamBeamSigmas_get_sigma33)( sigmas );
    SIXTRL_REAL_T const sigma_34 = NS(BeamBeamSigmas_get_sigma11)( sigmas );    
    SIXTRL_REAL_T const sigma_44 = NS(BeamBeamSigmas_get_sigma44)( sigmas );
    
    SIXTRL_REAL_T const sigma_11_p = sigma_11 + TWO * sigma_12 * s 
                                   + sigma_22 * s_squ;
                             
    SIXTRL_REAL_T const sigma_12_p = sigma_12 + sigma_22 * s;
    SIXTRL_REAL_T const sigma_13_p = sigma_13 + ( sigma_14 + sigma_23 ) * s 
                                   + sigma_24 * s_squ;
                             
    SIXTRL_REAL_T const sigma_14_p = sigma_14 + sigma_24 * s;
    
    SIXTRL_REAL_T const sigma_22_p = sigma_22;
    SIXTRL_REAL_T const sigma_23_p = sigma_23 + sigma_24 * s;
    SIXTRL_REAL_T const sigma_24_p = sigma_24;
    
    SIXTRL_REAL_T const sigma_33_p = sigma_33 + TWO * sigma_34 * s 
                                   + sigma_44 * s_squ;
                             
    SIXTRL_REAL_T const sigma_34_p = sigma_34 + sigma_44 * s;
    SIXTRL_REAL_T const sigma_44_p = sigma_44;
    
    SIXTRL_REAL_T const R = sigma_11_p - sigma_33_p;
    SIXTRL_REAL_T const W = sigma_11_p + sigma_33_p;
    SIXTRL_REAL_T const T = R * R + FOUR * sigma_13_p * sigma_13_p;
    
    SIXTRL_REAL_T const dS_R = TWO *     ( sigma_12 - sigma_34 ) 
                             + TWO * s * ( sigma_22 - sigma_44 );
                             
    SIXTRL_REAL_T const dS_W = TWO *     ( sigma_12 + sigma_34 ) 
                             + TWO * s * ( sigma_22 + sigma_44 );
                             
    SIXTRL_REAL_T const dS_sigma_13_p = sigma_14 + sigma_23 
                                      + TWO * s * sigma_24;
                                      
    SIXTRL_REAL_T const dS_T = TWO * R * dS_R 
                             + EIGHT * sigma_13_p * dS_sigma_13_p;
    
    SIXTRL_ASSERT( ( sigmas != 0 ) && 
                   ( ptr_result != 0 ) && ( ptr_deriv_result != 0 ) );
    
    NS(BeamBeamPropagatedSigmasResult_preset)( &result );
    NS(BeamBeamPropagatedSigmasResult_preset)( &deriv_result );
    
    if( ( T < treshold_singular ) && ( handle_sigularity ) )
    {
        SIXTRL_REAL_T const A = sigma_12_p - sigma_34_p;
        SIXTRL_REAL_T const B = sigma_22_p - sigma_44_p;
        SIXTRL_REAL_T const C = sigma_14_p - sigma_23_p;
        SIXTRL_REAL_T const D = sigma_24_p;
        
        SIXTRL_REAL_T const B_SIGN = ( B >= ZERO ) ? 1 : -1;
        SIXTRL_REAL_T const D_SIGN = ( D >= ZERO ) ? 1 : -1;
        
        SIXTRL_REAL_T const A_SQU_PLUS_C_SQU = A * A + C * C;
        SIXTRL_REAL_T const SQRT_A_SQU_C_SQU = sqrt( A_SQU_PLUS_C_SQU );
        
        SIXTRL_REAL_T const SQRT_A_SQU_C_SQU_POW_32 = 
            A_SQU_PLUS_C_SQU * SQRT_A_SQU_C_SQU;
        
        result.sigma_11_hat = 
        result.sigma_33_hat = ONE_HALF * W; 
        
        deriv_result.sigma_11_hat = 
        deriv_result.sigma_33_hat = ONE_HALF * dS_W;
            
        if( SQRT_A_SQU_C_SQU_POW_32 < treshold_singular )
        {
            SIXTRL_REAL_T cos2theta    = ONE;
            SIXTRL_REAL_T const FABS_D = ( D >= ZERO ) ? D : -D;
            
            if( FABS_D > treshold_singular )
            {
                SIXTRL_REAL_T const FABS_B    = ( B >= ZERO ) ? B : -B;
                cos2theta = FABS_B / sqrt( B * B + FOUR * D * D );                
            }
            
            result.cos_theta =  sqrt( ONE_HALF * ( ONE + cos2theta ) );
            result.sin_theta = B_SIGN * D_SIGN * 
                sqrt( ONE_HALF * ( ONE - cos2theta ) );
                
            deriv_result.cos_theta =
            deriv_result.sin_theta = ZERO;
        }
        else
        {
            SIXTRL_REAL_T const SIGN_A    = ( A >= ZERO ) ? ONE : -ONE;       
            SIXTRL_REAL_T const SIGN_C    = ( C >= ZERO ) ? ONE : -ONE;
            SIXTRL_REAL_T const cos2theta = ( SIGN_A * A ) / SQRT_A_SQU_C_SQU;
            
            SIXTRL_REAL_T FABS_SIN_THETA  = ZERO;
            SIXTRL_REAL_T COS_THETA       = ZERO;
            SIXTRL_REAL_T SIN_THETA       = ZERO;            
            
            SIXTRL_REAL_T dS_cos2theta = 
                SIGN_A * ( ONE_HALF * B / SQRT_A_SQU_C_SQU - 
                    A * ( A * B + TWO * C * D ) / 
                        ( TWO * SQRT_A_SQU_C_SQU_POW_32 ) );
            
            COS_THETA = result.cos_theta = 
                sqrt( ONE_HALF * ( ONE + cos2theta ) );
            
            SIN_THETA = result.sin_theta = 
                SIGN_A * SIGN_C * sqrt( ONE_HALF * ( ONE - cos2theta ) );
            
            FABS_SIN_THETA = ( SIN_THETA > ZERO ) ? SIN_THETA : -SIN_THETA;
                
            deriv_result.cos_theta = dS_cos2theta / ( FOUR * COS_THETA );
            
            deriv_result.sin_theta = ( FABS_SIN_THETA > treshold_singular )
                ? -ONE / ( FOUR * result.sin_theta ) * dS_cos2theta
                : D / ( TWO * A );
            
            deriv_result.sigma_11_hat += SIGN_A * SQRT_A_SQU_C_SQU;
            deriv_result.sigma_33_hat -= SIGN_A * SQRT_A_SQU_C_SQU;
        }
    }
    else
    {
        SIXTRL_REAL_T const SQRT_T    = sqrt( T );
        SIXTRL_REAL_T const SIGN_R    = ( R >= ZERO ) ? ONE : -ONE;
        SIXTRL_REAL_T const COS2THETA = SIGN_R * R / SQRT_T;
        SIXTRL_REAL_T const COS_THETA = sqrt( ONE_HALF * ( ONE + COS2THETA ) );
        
        SIXTRL_REAL_T const SIGN_SIGMA_13_P = 
            ( sigma_13_p >= ZERO ) ? ONE : -ONE;
        
        SIXTRL_REAL_T const SIN_THETA = 
            SIGN_R * SIGN_SIGMA_13_P * sqrt( ONE_HALF * ( ONE - COS2THETA ) );
        
        SIXTRL_REAL_T const FABS_SIN_THETA = 
            ( SIN_THETA >= ZERO ) ? SIN_THETA : -SIN_THETA;
            
        SIXTRL_REAL_T const dS_cos2theta = SIGN_R * ( dS_R / SQRT_T - 
            dS_T * R / ( TWO * SQRT_T * SQRT_T * SQRT_T ) );
        
        SIXTRL_REAL_T const dS_costheta = dS_cos2theta / ( FOUR * COS_THETA );
            
        result.cos_theta = COS_THETA;
        result.sin_theta = SIN_THETA;
        
        result.sigma_11_hat = ONE_HALF * ( W + SIGN_R * SQRT_T );
        result.sigma_33_hat = ONE_HALF * ( W - SIGN_R * SQRT_T );
            
        deriv_result.cos_theta = dS_costheta;
        deriv_result.sin_theta = ( ( !handle_sigularity ) ||
            ( FABS_SIN_THETA >= treshold_singular ) )
            ? -dS_cos2theta / ( FOUR * SIN_THETA )
            : ( sigma_14_p + sigma_23_p ) / R;
            
        deriv_result.sigma_11_hat = 
            ONE_HALF * ( dS_W + SIGN_R * dS_T / SQRT_T  );
            
        deriv_result.sigma_33_hat = 
            ONE_HALF * ( dS_W - SIGN_R * dS_T / SQRT_T  );
    }
    
    *ptr_result       = result;
    *ptr_deriv_result = deriv_result;
    
    return ret;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(BeamBeam_complex_error_function)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT result_real,
        SIXTRL_REAL_T* SIXTRL_RESTRICT result_imag,
        SIXTRL_REAL_T const input_real, SIXTRL_REAL_T const input_imag )
{
    /**
    this function calculates the double precision complex error function based on the
    algorithm of the FORTRAN function written at CERN by K. Koelbig, Program C335, 1970.
    See also M. Bassetti and G.A. Erskine, "Closed expression for the electric field of a 
    two-dimensional Gaussian charge density", CERN-ISR-TH/80-06;
    */

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const A_CONSTANT = 
        ( SIXTRL_REAL_T )1.12837916709551L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const xLim = ( SIXTRL_REAL_T )5.33L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const yLim = ( SIXTRL_REAL_T )4.29L;
    
    SIXTRL_REAL_T const x = ( input_real >= ZERO ) ? input_real : -input_real;
    SIXTRL_REAL_T const y = ( input_imag >= ZERO ) ? input_imag : -input_imag;
    
    SIXTRL_REAL_T Wx = ZERO;
    SIXTRL_REAL_T Wy = ZERO;
    
    if( ( y < yLim ) && ( x < xLim ) )
    {
        SIXTRL_REAL_T const x_sc = x / xLim;
        SIXTRL_REAL_T const q  = ( ONE - y / yLim ) * sqrt( ONE - x_sc * x_sc );
        SIXTRL_REAL_T const h  = ONE / ( ( SIXTRL_REAL_T )3.2L * q );
        
        SIXTRL_INT64_T  ii = 0;
        SIXTRL_INT64_T  const nc = ( SIXTRL_INT64_T )7u + ( SIXTRL_INT64_T )( 
            q * ( ( SIXTRL_REAL_T )23.0L ) );
        
        SIXTRL_REAL_T       xl = pow( h, ( ONE - nc ) );
        SIXTRL_REAL_T const xh = y + ONE_HALF * h;
        SIXTRL_REAL_T const yh = x;
        
        SIXTRL_INT64_T const nu = ( SIXTRL_INT64_T )10u + ( SIXTRL_INT64_T )(
            q * ( ( SIXTRL_REAL_T )21.0L ) );
        
        SIXTRL_REAL_T Rx[ 33 ];
        SIXTRL_REAL_T Ry[ 33 ];
        
        SIXTRL_REAL_T Sx = ZERO;
        SIXTRL_REAL_T Sy = ZERO;
        
        SIXTRL_ASSERT( nu < 33 );
        
        Rx[ nu ] = ZERO;
        Ry[ nu ] = ZERO;
        
        for( ii = nu ; ii > 0 ; --ii )
        {
            SIXTRL_REAL_T const Tx = xh + ii * Rx[ ii ];
            SIXTRL_REAL_T const Ty = yh - ii * Ry[ ii ];
            SIXTRL_REAL_T const Tn = Tx * Tx + Ty * Ty;
            
            Rx[ ii - 1 ] = ONE_HALF * Tx / Tn;
            Ry[ ii - 1 ] = ONE_HALF * Ty / Tn;            
        }
        
        for( ii = nc ; ii > 0 ; --ii )
        {
            SIXTRL_REAL_T const Saux = Sx + xl;
            
            Sx  = Rx[ ii - 1 ] * Saux - Ry[ ii - 1 ] * Sy;
            Sy *= Rx[ ii - 1 ];
            Sy += Ry[ ii - 1 ] * Saux;
            xl *= h;
        }
        
        Wx = A_CONSTANT * Sx;
        Wy = A_CONSTANT * Sy;
    }
    else
    {
        SIXTRL_REAL_T const xh = y;
        SIXTRL_REAL_T const yh = x;
        
        SIXTRL_REAL_T Rx[ 10 ] =
        { 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO             
        };
            
        SIXTRL_REAL_T Ry[ 10 ] =
        { 
            ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO             
        };
        
        SIXTRL_INT64_T ii = 0;
        SIXTRL_INT64_T const nu = 9;
                
        for( ii = nu ; ii > 0 ; --ii )
        {
            SIXTRL_REAL_T const Tx = xh + ii * Rx[ 0 ];
            SIXTRL_REAL_T const Ty = yh - ii * Ry[ 0 ];
            SIXTRL_REAL_T const Tn = Tx * Tx + Ty * Ty;
            
            Rx[ 0 ] = ONE_HALF * Tx / Tn;
            Ry[ 0 ] = ONE_HALF * Ty / Tn;
        }
        
        Wx = A_CONSTANT * Rx[0];
        Wy = A_CONSTANT * Ry[0];
    }
    
    if( ( input_imag >= ZERO ) && ( input_imag < ( SIXTRL_REAL_T )1e-16L ) )
    {
        Wx = exp( -x * x );
    }
    
    if( input_imag < ZERO )
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO = ( SIXTRL_REAL_T )2.0L;
        
        Wx =  TWO * exp( y * y - x * x ) * cos( TWO * x * y) - Wx;
        Wy = -TWO * exp( y * y - x * x ) * sin( TWO * x * y) - Wy;
        
        if( input_real > ZERO )
        {
            Wy = -Wy;            
        }
    }
    else if( input_real < ZERO ) 
    {
        Wy = -Wy;
    }
    
    SIXTRL_ASSERT( result_real != 0 );
    SIXTRL_ASSERT( result_imag != 0 );
    
    *result_real = Wx;
    *result_imag = Wy;
    
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
    NS(BeamBeam_get_transverse_fields)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gx_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gy_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y, 
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const min_sigma_diff )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN  )0u;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const EPSILON_0 = 
        ( SIXTRL_REAL_T )8.854187817620e-12L;
        
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const PI = 
        ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751L;
    
    SIXTRL_REAL_T const delta_sigma = sigma_x - sigma_y;
    
    SIXTRL_REAL_T const abs_delta_sigma = 
        ( delta_sigma >= ZERO ) ? delta_sigma : -delta_sigma;
    
    SIXTRL_REAL_T Ex = ZERO;
    SIXTRL_REAL_T Ey = ZERO;
        
    if( abs_delta_sigma < min_sigma_diff )
    {
        SIXTRL_REAL_T const sigma = ONE_HALF * ( sigma_x + sigma_y );
        
        ret = NS(BeamBeam_get_transverse_fields_gauss_round)(
            &Ex, &Ey, x, y, sigma, ZERO, ZERO );
        
        if( ( ret == 0 ) && ( gx_component != 0 ) && ( gy_component != 0 ) )
        {
            SIXTRL_REAL_T const TWO_SIGMA_SQU = TWO * sigma * sigma;
            SIXTRL_REAL_T const FACTOR = TWO_SIGMA_SQU * PI * EPSILON_0;
            
            SIXTRL_REAL_T const x_squ = x * x;
            SIXTRL_REAL_T const y_squ = y * y;
            SIXTRL_REAL_T const r_squ = x_squ + y_squ;
            
            SIXTRL_ASSERT( ( gx_component != 0 ) && ( gy_component != 0 ) );
            
            *gx_component = ( x_squ * exp( -r_squ / TWO_SIGMA_SQU ) *
                ( y * Ey - x * Ex + ONE / FACTOR ) ) / ( TWO * r_squ );
                
            *gy_component = ( y_squ * exp( -r_squ / TWO_SIGMA_SQU ) *
                ( x * Ex - y * Ey + ONE / FACTOR ) ) / ( TWO * r_squ );
        }
    }
    else 
    {
        SIXTRL_REAL_T const sigma_11 = sigma_x * sigma_x;
        SIXTRL_REAL_T const sigma_33 = sigma_y * sigma_y;
        
        ret = NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
            &Ex, &Ey, x, y, sigma_x, sigma_y, ZERO, ZERO );
        
        if( ( ret == 0 ) && ( gx_component != 0 ) && ( gy_component != 0 ) )
        {
            SIXTRL_REAL_T const x_squ  = x * x;
            SIXTRL_REAL_T const y_squ  = y * y;
            SIXTRL_REAL_T const FACTOR = TWO * PI * EPSILON_0;
            SIXTRL_REAL_T const DELTA_SIGMA_11_33 = sigma_11 - sigma_33;
            SIXTRL_REAL_T const TEMP = x_squ / sigma_11 + y_squ / sigma_33;
            
            *gx_component = ( x * Ex + y * Ey + 
                ( ( sigma_y / sigma_x ) * exp( TEMP ) - ONE ) / FACTOR ) /
                    ( TWO * DELTA_SIGMA_11_33 );
                    
            *gy_component = ( x * Ex + y * Ey +
                ( ( sigma_x / sigma_y ) * exp( TEMP ) - ONE ) / FACTOR ) /
                    ( TWO * DELTA_SIGMA_11_33 );            
        }
    }
    
    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );
    
    *ex_component = Ex;
    *ey_component = Ey;
    
    return ret;
}
  
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
NS(BeamBeam_get_transverse_fields_gauss_round)(
    SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component, 
    SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_REAL_T const sigma, 
    SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y )
{
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const EPSILON_0 = 
        ( SIXTRL_REAL_T )8.854187817620e-12L;
        
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const PI = 
        ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751L;
    
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0u;
        
    SIXTRL_REAL_T const diff_x    = ( x - delta_x );
    SIXTRL_REAL_T const diff_y    = ( y - delta_y );    
    SIXTRL_REAL_T const r_squ     = diff_x * diff_x + diff_y * diff_y;
    
    SIXTRL_REAL_T const temp = ( r_squ >= ( SIXTRL_REAL_T )1e-20 )
        ? sqrt( r_squ ) / ( TWO * PI * EPSILON_0 * sigma )
        : ( ONE - exp( -ONE_HALF * r_squ / ( sigma * sigma ) ) ) / 
            ( TWO * PI * EPSILON_0 * r_squ );
    
    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );
    
    *ex_component = temp * diff_x;
    *ey_component = temp * diff_y;
    
    return ret;
}
        
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        
SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN 
NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
    SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
    SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
    SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0u;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const EPSILON_0 = 
        ( SIXTRL_REAL_T )8.854187817620e-12L;
        
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const SQRT_PI = 
        ( SIXTRL_REAL_T )1.7724538509055160272981674833411451827975494561224L;
    
    SIXTRL_REAL_T const diff_x      = x - delta_x;
    SIXTRL_REAL_T const abs_diff_x  = ( diff_x >= ZERO ) ? diff_x : -diff_x;
    
    SIXTRL_REAL_T const diff_y      = y - delta_y;
    SIXTRL_REAL_T const abs_diff_y  = ( diff_y >= ZERO ) ? diff_y : -diff_y;
    
    SIXTRL_REAL_T const ETA_ERRFUN_SCALE_FACTOR = exp( -ONE_HALF * 
            ( ( abs_diff_x * abs_diff_x ) / ( sigma_x * sigma_x ) + 
              ( abs_diff_y * abs_diff_y ) / ( sigma_y * sigma_y ) ) );
    
    SIXTRL_ASSERT( ( sigma_x * sigma_x ) >= ( SIXTRL_REAL_T )1e-16L );
    SIXTRL_ASSERT( ( sigma_y * sigma_y ) >= ( SIXTRL_REAL_T )1e-16L );
    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );
    
    if( sigma_x > sigma_y )
    {
        SIXTRL_REAL_T const S = 
            sqrt( TWO * ( sigma_x * sigma_x - sigma_y * sigma_y ) );
        
        SIXTRL_REAL_T const factBE = ONE / ( TWO * EPSILON_0 * SQRT_PI * S );
        
        SIXTRL_REAL_T eta_be_x  = ( sigma_y / sigma_x * abs_diff_x ) / S;
        SIXTRL_REAL_T eta_be_y  = ( sigma_x / sigma_y * abs_diff_y ) / S;
                
        SIXTRL_REAL_T zeta_be_x = abs_diff_x / S;
        SIXTRL_REAL_T zeta_be_y = abs_diff_y / S;
        
        NS(BeamBeam_complex_error_function)( 
            &zeta_be_x, &zeta_be_y, zeta_be_x, zeta_be_y );
        
        NS(BeamBeam_complex_error_function)(
            &eta_be_x, &eta_be_y, eta_be_x, eta_be_y );
        
        eta_be_x *= ETA_ERRFUN_SCALE_FACTOR;
        eta_be_y *= ETA_ERRFUN_SCALE_FACTOR;
        
        *ex_component = ( zeta_be_x - eta_be_x ) * factBE;
        *ey_component = ( zeta_be_y - eta_be_y ) * factBE;
    }
    else if( sigma_x < sigma_y )
    {
        SIXTRL_REAL_T const S = 
            sqrt( TWO * ( sigma_y * sigma_y - sigma_x * sigma_x ) );
        
        SIXTRL_REAL_T const factBE =  ONE / ( TWO * EPSILON_0 * SQRT_PI * S );
        
        SIXTRL_REAL_T eta_be_x  = ( sigma_x / sigma_y * abs_diff_y ) / S;
        SIXTRL_REAL_T eta_be_y  = ( sigma_y / sigma_x * abs_diff_x ) / S;
                
        SIXTRL_REAL_T zeta_be_x = abs_diff_y / S;
        SIXTRL_REAL_T zeta_be_y = abs_diff_x / S;
        
        NS(BeamBeam_complex_error_function)( 
            &zeta_be_x, &zeta_be_y, zeta_be_x, zeta_be_y );
        
        NS(BeamBeam_complex_error_function)(
            &eta_be_x, &eta_be_y, eta_be_x, eta_be_y );
        
        eta_be_x *= ETA_ERRFUN_SCALE_FACTOR;
        eta_be_y *= ETA_ERRFUN_SCALE_FACTOR;
        
        *ex_component = ( zeta_be_x - eta_be_x ) * factBE;
        *ey_component = ( zeta_be_y - eta_be_y ) * factBE;
    }
    else
    {
        /* ?????? */
        *ex_component = *ey_component = ONE / ZERO;
    }
    
    if( delta_x < ZERO ) *ex_component = -( *ex_component );
    if( delta_y < ZERO ) *ey_component = -( *ey_component );
    
    return ret;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_beam_beam_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0u;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* x_slices_st_begin = 
        NS(BeamBeam_get_const_x_slices_star)( beam_beam );
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* y_slices_st_begin =
        NS(BeamBeam_get_const_y_slices_star)( beam_beam );
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* sigma_slices_st_begin =
        NS(BeamBeam_get_const_sigma_slices_star)( beam_beam );
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* n_part_per_slice_begin =
        NS(BeamBeam_get_const_n_part_per_slice)( beam_beam );
        
    SIXTRL_REAL_T x_st      = ZERO;
    SIXTRL_REAL_T y_st      = ZERO;
    SIXTRL_REAL_T px_st     = ZERO;
    SIXTRL_REAL_T py_st     = ZERO;
    SIXTRL_REAL_T sigma_st  = ZERO;
    SIXTRL_REAL_T delta_st  = ZERO;
        
    SIXTRL_REAL_T const p0c = NS(Particles_get_p0c_value)( particles, ii );
    SIXTRL_REAL_T const q0  = NS(Particles_get_q0_value)(  particles, ii );
        
    SIXTRL_REAL_T const q_part = NS(BeamBeam_get_q_part)( beam_beam );
    SIXTRL_REAL_T const min_sigma_diff = 
        NS(BeamBeam_get_min_sigma_diff)( beam_beam );
    
    SIXTRL_REAL_T const treshold_singular = 
        NS(BeamBeam_get_treshold_singular)( beam_beam );
    
    NS(block_num_elements_t) const  NUM_SLICES = 
        NS(BeamBeam_get_num_of_slices)( beam_beam );
        
    NS(block_num_elements_t) jj = 0;
        
    #if defined( _GPUCODE ) && !defined( _CUDACC__ )
    
    NS(BeamBeamBoostData) const* boost_data  = 0;
    NS(BeamBeamSigmas)    const* sigmas_data = 0;
    
    SIXTRL_GLOBAL_DEC NS(BeamBeamBoostData) const* ptr_g_boost = 
        NS(BeamBeam_get_const_ptr_boost_data)( beam_beam );
    
    SIXTRL_GLOBAL_DEC NS(BeamBeamSigmas) const* ptr_g_sigmas_matrix = 
        NS(BeamBeam_get_const_ptr_sigmas_matrix)( beam_beam );
        
    NS(BeamBeamBoostData) temp_boost;
    NS(BeamBeamSigmas)    temp_sigmas;
        
    if( ptr_g_boost != 0 )
    {
        temp_boost = *ptr_g_boost;        
    }
    else
    {
        NS(BeamBeamBoostData_preset)( &temp_boost );
    }
    
    if( ptr_g_sigmas_matrix != 0 )
    {
        temp_sigmas = *ptr_g_sigmas_matrix;
    }
    else
    {
        NS(BeamBeamSigmas_preset)( &temp_sigmas );
    }
            
    boost_data  = &temp_boost;
    sigmas_data = &temp_sigmas;
    
    #else /* !defined( _GPUCODE ) || defined( _CUDACC__ ) */
    
    NS(BeamBeamBoostData) const* boost_data  = 
        NS(BeamBeam_get_const_ptr_boost_data)( beam_beam );
        
    NS(BeamBeamSigmas) const* sigmas_data =
        NS(BeamBeam_get_const_ptr_sigmas_matrix)( beam_beam );
    
    #endif /* defined( _GPUCODE ) && !defined( _CUDACC__ ) */
        
    SIXTRL_ASSERT( ( boost_data             != 0 ) && 
                   ( sigmas_data            != 0 ) &&
                   ( x_slices_st_begin      != 0 ) &&
                   ( y_slices_st_begin      != 0 ) &&
                   ( sigma_slices_st_begin  != 0 ) &&
                   ( n_part_per_slice_begin != 0 ) );
    
    ret = NS(BeamBeam_boost_particle)( particles, ii, boost_data );
    
    x_st      = NS(Particles_get_x_value)( particles, ii );
    y_st      = NS(Particles_get_y_value)( particles, ii );
    px_st     = NS(Particles_get_px_value)( particles, ii );
    py_st     = NS(Particles_get_py_value)( particles, ii );
    sigma_st  = NS(Particles_get_sigma_value)( particles, ii );
    delta_st  = NS(Particles_get_delta_value)( particles, ii );
    
    for( ; jj < NUM_SLICES ; ++jj )
    {
        SIXTRL_REAL_T const sigma_sl_st = sigma_slices_st_begin[ jj ];
        SIXTRL_REAL_T const x_sl_st     = x_slices_st_begin[ jj ];
        SIXTRL_REAL_T const y_sl_st     = y_slices_st_begin[ jj ];        
        
        SIXTRL_REAL_T const ksl         = 
            n_part_per_slice_begin[ jj ] * q_part * q0 / p0c;
                
        SIXTRL_REAL_T const S = ONE_HALF * ( sigma_st - sigma_sl_st );
        
        NS(BeamBeamPropagatedSigmasResult) result;
        NS(BeamBeamPropagatedSigmasResult) dS_result;
        
        SIXTRL_TRACK_RETURN const ret_sigma = 
            NS(BeamBeam_propagate_sigma_matrix)( &result, &dS_result, 
                sigmas_data, S, treshold_singular, 1 );
        
        SIXTRL_REAL_T const x_bar_st = x_st + px_st * S - x_sl_st;
        SIXTRL_REAL_T const y_bar_st = y_st + py_st * S - y_sl_st;
        
        SIXTRL_REAL_T const x_bar_hat_st =  x_bar_st * result.cos_theta +
                                            y_bar_st * result.sin_theta;
        
        SIXTRL_REAL_T const y_bar_hat_st = -x_bar_st * result.sin_theta +
                                            y_bar_st * result.cos_theta;
                                            
        SIXTRL_REAL_T const dS_x_bar_hat_st = x_bar_st * dS_result.cos_theta 
                                            + y_bar_st * dS_result.sin_theta;
        
        SIXTRL_REAL_T const dS_y_bar_hat_st = -x_bar_st * dS_result.sin_theta +
                                               y_bar_st * dS_result.cos_theta;
        
        SIXTRL_REAL_T Ex = ZERO;
        SIXTRL_REAL_T Ey = ZERO;
        SIXTRL_REAL_T Gx = ZERO; 
        SIXTRL_REAL_T Gy = ZERO;
        
        SIXTRL_TRACK_RETURN const ret_get_transverse_fields = 
            NS(BeamBeam_get_transverse_fields)( &Ex, &Ey, &Gx, &Gy, 
                x_bar_hat_st, y_bar_hat_st, 
                sqrt( result.sigma_11_hat ), sqrt( result.sigma_33_hat ),
                min_sigma_diff );
            
        SIXTRL_REAL_T const Fx_hat_st = ksl * Ex;
        SIXTRL_REAL_T const Fy_hat_st = ksl * Ey;
        SIXTRL_REAL_T const Gx_hat_st = ksl * Gx;
        SIXTRL_REAL_T const Gy_hat_st = ksl * Gy;
        
        SIXTRL_REAL_T const Fx_st = Fx_hat_st * result.cos_theta - 
                                    Fy_hat_st * result.sin_theta;
                                    
        SIXTRL_REAL_T const Fy_st = Fx_hat_st * result.sin_theta +
                                    Fy_hat_st * result.cos_theta;
        
        SIXTRL_REAL_T const Fz_st = ONE_HALF * (
            Fx_hat_st * dS_x_bar_hat_st        + 
            Fy_hat_st * dS_y_bar_hat_st        +
            Gx_hat_st * dS_result.sigma_11_hat +
            Gy_hat_st * dS_result.sigma_33_hat );
        
        
        delta_st += Fz_st + ONE_HALF * (
                    Fx_st * ( px_st + ONE_HALF * Fx_st ) + 
                    Fy_st * ( py_st + ONE_HALF * Fy_st ) );
        
        x_st  -= S * Fx_st;
        y_st  -= S * Fy_st;
        
        px_st += Fx_st;
        py_st += Fy_st;
        
        ret |= ret_sigma;
        ret |= ret_get_transverse_fields;
    }
    
    NS(Particles_set_x_value)(     particles, ii, x_st  );
    NS(Particles_set_y_value)(     particles, ii, y_st  );
    NS(Particles_set_px_value)(    particles, ii, px_st );
    NS(Particles_set_py_value)(    particles, ii, py_st );
    NS(Particles_set_sigma_value)( particles, ii, sigma_st );
    NS(Particles_set_delta_value)( particles, ii, delta_st );
    
    ret |= NS(BeamBeam_inv_boost_particle)( particles, ii, boost_data );
    
    return ret;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN 
NS(Track_range_of_particles_over_beam_element)(    
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) index,
    NS(block_num_elements_t) const index_end,    
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* SIXTRL_RESTRICT be_block_it )
{
    int ret = 0;
    
    #if !defined( _GPUCODE )
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( be_block_it );
    
    SIXTRL_GLOBAL_DEC void const* ptr_beam_element_begin =
        NS(BlockInfo_get_const_ptr_begin)( be_block_it );
        
    #else /* !defined( _GPUCODE ) */
    
    NS(BlockInfo) const info    = *be_block_it;
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( &info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_beam_element_begin =
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
        
        case NS(BLOCK_TYPE_CAVITY):
        {
            typedef NS(Cavity) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_cavity_particle)( particles, index, &be );
            }
            
            break;
        }
        
        case NS(BLOCK_TYPE_ALIGN):
        {
            typedef NS(Align) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_align_particle)( particles, index, &be );
            }
            
            break;
        }
        
        case NS(BLOCK_TYPE_BEAM_BEAM):
        {
            typedef NS(BeamBeam) be_t;
            typedef SIXTRL_GLOBAL_DEC be_t const* g_be_ptr_t;
            
            be_t const be = *( ( g_be_ptr_t )ptr_beam_element_begin );
            
            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_beam_beam_particle)( particles, index, &be );
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
