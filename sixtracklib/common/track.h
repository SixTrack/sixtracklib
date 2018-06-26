#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <complex.h>
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

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_beam_beam_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    const NS(BeamBeam) *const SIXTRL_RESTRICT beam_beam );
        
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
#include "sixtracklib/common/impl/beam_beam_element_6d.h"

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
