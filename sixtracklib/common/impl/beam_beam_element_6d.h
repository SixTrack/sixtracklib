#ifndef SIXTRACKL_COMMON_IMPL_BEAM_BEAM_ELEMENT_6D_H__
#define SIXTRACKL_COMMON_IMPL_BEAM_BEAM_ELEMENT_6D_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/blocks.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

struct NS(Particles);
struct NS(BeamBeamBoostData);
struct NS(BeamBeamSigmas);
struct NS(BeamBeamPropagatedSigmasResult);

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_boost_particle)(
    struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const index,
    const struct NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_inv_boost_particle)(
    struct NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const index,
    const struct NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_propagate_sigma_matrix)(
        struct NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT result,
        struct NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ds_result,
        const struct NS(BeamBeamSigmas) *const sigma_matrix,
        SIXTRL_REAL_T  const s,
        SIXTRL_REAL_T  const treshold_singular,
        SIXTRL_INT64_T const handle_sigularity );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_get_transverse_fields)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gx_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gy_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const min_sigma_diff );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_get_transverse_fields_gauss_round)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma,
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined ( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined ( _GPUCODE ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/impl/particles_api.h"
    #include "sixtracklib/common/impl/faddeeva.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(BeamBeam_boost_particle)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii,
    const struct NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost )
{
    int ret = 0;

    #if !defined( NDEBUG ) && !defined( __CUDACC__ )
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO    = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const MIN_EPS = ( SIXTRL_REAL_T )1e-12L; /* ?? */
    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE     = ( SIXTRL_REAL_T )1.0L;

    SIXTRL_REAL_T px    = NS(Particles_get_px_value)(    p, ii );
    SIXTRL_REAL_T py    = NS(Particles_get_py_value)(    p, ii );
    SIXTRL_REAL_T delta = NS(Particles_get_delta_value)( p, ii );

    SIXTRL_REAL_T const cos_phi   = NS(BeamBeamBoostData_get_cphi)( boost );
    SIXTRL_REAL_T const sin_phi   = NS(BeamBeamBoostData_get_sphi)( boost );
    SIXTRL_REAL_T const tan_phi   = NS(BeamBeamBoostData_get_tphi)( boost );
    SIXTRL_REAL_T const sin_alpha = NS(BeamBeamBoostData_get_salpha)( boost );
    SIXTRL_REAL_T const cos_alpha = NS(BeamBeamBoostData_get_calpha)( boost );

    SIXTRL_REAL_T const sin_alpha_tan_phi = sin_alpha * tan_phi;
    SIXTRL_REAL_T const cos_alpha_tan_phi = cos_alpha * tan_phi;
    SIXTRL_REAL_T const sin_alpha_sin_phi = sin_alpha * sin_phi;
    SIXTRL_REAL_T const cos_alpha_sin_phi = cos_alpha * sin_phi;
    SIXTRL_REAL_T const inv_cos_phi       = ONE / cos_phi;

    SIXTRL_REAL_T const delta_plus_one = delta + ONE;

    SIXTRL_REAL_T const h_sqrt_arg =
        delta_plus_one * delta_plus_one - px * px - py * py;

    SIXTRL_REAL_T const h = delta_plus_one - sqrt( h_sqrt_arg );

    /* --------------------------------------------------------------------- */

    SIXTRL_REAL_T const delta_star  =
        delta   - px * cos_alpha_tan_phi
                - py * sin_alpha_tan_phi
                + h  * tan_phi * tan_phi;

    SIXTRL_REAL_T const delta_star_plus_one = delta_star + ONE;

    SIXTRL_REAL_T const px_star = inv_cos_phi * ( px - h * cos_alpha_tan_phi );
    SIXTRL_REAL_T const py_star = inv_cos_phi * ( py - h * sin_alpha_tan_phi );

    SIXTRL_REAL_T const pz_star_squ =
        delta_star_plus_one * delta_star_plus_one
        - px_star * px_star
        - py_star * py_star;

    SIXTRL_REAL_T const pz_star     = sqrt( pz_star_squ );
    SIXTRL_REAL_T const hx_star     = px_star / pz_star;
    SIXTRL_REAL_T const hy_star     = py_star / pz_star;
    SIXTRL_REAL_T const hsigma_star = ONE - ( delta_star_plus_one / pz_star );

    SIXTRL_REAL_T const L11         = ONE + hx_star * cos_alpha_sin_phi;
    SIXTRL_REAL_T const L12         = hx_star * sin_alpha_sin_phi;
    SIXTRL_REAL_T const L13         = cos_alpha_tan_phi;

    SIXTRL_REAL_T const L21         = hy_star * cos_alpha_sin_phi;
    SIXTRL_REAL_T const L22         = ONE + hy_star * sin_alpha_sin_phi;
    SIXTRL_REAL_T const L23         = sin_alpha_tan_phi;

    SIXTRL_REAL_T const L31         = hsigma_star * cos_alpha_sin_phi;
    SIXTRL_REAL_T const L32         = hsigma_star * sin_alpha_sin_phi;
    SIXTRL_REAL_T const L33         = inv_cos_phi;

    SIXTRL_REAL_T const x     = NS(Particles_get_x_value)(     p, ii );
    SIXTRL_REAL_T const y     = NS(Particles_get_y_value)(     p, ii );
    SIXTRL_REAL_T const sigma = NS(Particles_get_sigma_value)( p, ii );

    SIXTRL_REAL_T const x_star     = L11 * x + L12 * y + L13 * sigma;
    SIXTRL_REAL_T const y_star     = L21 * x + L22 * y + L23 * sigma;
    SIXTRL_REAL_T const sigma_star = L31 * x + L32 * y + L33 * sigma;

    #if !defined( NDEBUG ) && !defined( __CUDACC__ )

    SIXTRL_ASSERT( ( ( cos_phi > ZERO ) && (  cos_phi > MIN_EPS ) ) ||
                   ( ( cos_phi < ZERO ) && ( -cos_phi > MIN_EPS ) ) );

    SIXTRL_ASSERT( ( ( pz_star > ZERO ) && (  pz_star > MIN_EPS ) ) ||
                   ( ( pz_star < ZERO ) && ( -pz_star > MIN_EPS ) ) );

    SIXTRL_ASSERT( h_sqrt_arg  > ZERO );
    SIXTRL_ASSERT( pz_star_squ > ZERO );

    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */

    NS(Particles_set_x_value)(     p, ii, x_star );
    NS(Particles_set_y_value)(     p, ii, y_star );
    NS(Particles_set_sigma_value)( p, ii, sigma_star );
    NS(Particles_set_px_value)(    p, ii, px_star );
    NS(Particles_set_py_value)(    p, ii, py_star );
    NS(Particles_set_delta_value)( p, ii, delta_star );

    return ret;
}

SIXTRL_INLINE int NS(BeamBeam_inv_boost_particle)(
    struct NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii,
    const struct NS(BeamBeamBoostData) *const SIXTRL_RESTRICT boost )
{
    int ret = 0;

    #if !defined( NDEBUG ) && !defined( __CUDACC__ )
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO    = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const MIN_EPS = ( SIXTRL_REAL_T )1e-12L; /* ?? */
    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0L;

    SIXTRL_REAL_T px_star    = NS(Particles_get_px_value)(    p, ii );
    SIXTRL_REAL_T py_star    = NS(Particles_get_py_value)(    p, ii );
    SIXTRL_REAL_T delta_star = NS(Particles_get_delta_value)( p, ii );

    SIXTRL_REAL_T const delta_star_plus_one = delta_star + ONE;
    SIXTRL_REAL_T const pz_star_squ =
        delta_star_plus_one * delta_star_plus_one
        - px_star * px_star
        - py_star * py_star;

    SIXTRL_REAL_T const pz_star     = sqrt( pz_star_squ );
    SIXTRL_REAL_T const hx_star     = px_star / pz_star;
    SIXTRL_REAL_T const hy_star     = py_star / pz_star;
    SIXTRL_REAL_T const hsigma_star = ONE - ( delta_star_plus_one / pz_star );

    SIXTRL_REAL_T const cos_phi   = NS(BeamBeamBoostData_get_cphi)( boost );
    SIXTRL_REAL_T const sin_phi   = NS(BeamBeamBoostData_get_sphi)( boost );
    SIXTRL_REAL_T const tan_phi   = NS(BeamBeamBoostData_get_tphi)( boost );
    SIXTRL_REAL_T const sin_alpha = NS(BeamBeamBoostData_get_salpha)( boost );
    SIXTRL_REAL_T const cos_alpha = NS(BeamBeamBoostData_get_calpha)( boost );

    SIXTRL_REAL_T const sin_alpha_tan_phi = sin_alpha * tan_phi;
    SIXTRL_REAL_T const cos_alpha_tan_phi = cos_alpha * tan_phi;

    SIXTRL_REAL_T const sin_alpha_sin_phi = sin_alpha * sin_phi;
    SIXTRL_REAL_T const cos_alpha_sin_phi = cos_alpha * sin_phi;
    SIXTRL_REAL_T const inv_cos_phi       = ONE / cos_phi;

    SIXTRL_REAL_T const det_L = inv_cos_phi + tan_phi *
        ( hx_star * cos_alpha + hy_star * sin_alpha - hsigma_star * sin_phi );

    SIXTRL_REAL_T const inv_det_L = ONE / det_L;


    SIXTRL_REAL_T const L11_inv = inv_det_L * ( inv_cos_phi +
        sin_alpha_tan_phi * ( hy_star - hsigma_star * sin_alpha_sin_phi ) );

    SIXTRL_REAL_T const L12_inv = inv_det_L * (
        sin_alpha_tan_phi * ( hsigma_star* cos_alpha_sin_phi - hx_star ) );

    SIXTRL_REAL_T const L13_inv = -inv_det_L * tan_phi * ( cos_alpha -
            sin_alpha_sin_phi * ( hx_star * sin_alpha - hy_star * cos_alpha ) );


    SIXTRL_REAL_T const L21_inv = inv_det_L * (
        cos_alpha_tan_phi * ( hsigma_star * sin_alpha_sin_phi - hy_star ) );

    SIXTRL_REAL_T const L22_inv = inv_det_L * ( inv_cos_phi +
        cos_alpha_tan_phi * ( hx_star - hsigma_star * cos_alpha_sin_phi ) );

    SIXTRL_REAL_T const L23_inv = -inv_det_L * tan_phi * ( sin_alpha +
        cos_alpha_sin_phi * ( hx_star * sin_alpha - hy_star * cos_alpha ) );


    SIXTRL_REAL_T const L31_inv = -inv_det_L * hsigma_star * cos_alpha_sin_phi;
    SIXTRL_REAL_T const L32_inv = -inv_det_L * hsigma_star * sin_alpha_sin_phi;
    SIXTRL_REAL_T const L33_inv =  inv_det_L *
        ( ONE + hx_star * cos_alpha_sin_phi + hy_star * sin_alpha_sin_phi );

    SIXTRL_REAL_T const x_star     = NS(Particles_get_x_value)(     p, ii );
    SIXTRL_REAL_T const y_star     = NS(Particles_get_y_value)(     p, ii );
    SIXTRL_REAL_T const sigma_star = NS(Particles_get_sigma_value)( p, ii );

    SIXTRL_REAL_T const x =
        L11_inv * x_star + L12_inv * y_star + L13_inv * sigma_star;

    SIXTRL_REAL_T const y =
        L21_inv * x_star + L22_inv * y_star + L23_inv * sigma_star;

    SIXTRL_REAL_T const sigma =
        L31_inv * x_star + L32_inv * y_star + L33_inv * sigma_star;

    SIXTRL_REAL_T const h =
        ( delta_star_plus_one - pz_star ) * cos_phi * cos_phi;

    SIXTRL_REAL_T const px    = px_star * cos_phi + h * cos_alpha_tan_phi;
    SIXTRL_REAL_T const py    = py_star * cos_phi + h * sin_alpha_tan_phi;
    SIXTRL_REAL_T const delta =
        delta_star + px * cos_alpha_tan_phi
                   + py * sin_alpha_tan_phi
                   - h * tan_phi * tan_phi;

    #if !defined( NDEBUG ) && !defined( __CUDACC__ )

    SIXTRL_ASSERT( ( ( cos_phi > ZERO ) && (  cos_phi > MIN_EPS ) ) ||
                   ( ( cos_phi < ZERO ) && ( -cos_phi > MIN_EPS ) ) );

    SIXTRL_ASSERT( ( ( pz_star > ZERO ) && (  pz_star > MIN_EPS ) ) ||
                   ( ( pz_star < ZERO ) && ( -pz_star > MIN_EPS ) ) );

    SIXTRL_ASSERT( ( ( det_L   > ZERO ) && (  det_L   > MIN_EPS ) ) ||
                   ( ( det_L   < ZERO ) && ( -det_L   > MIN_EPS ) ) );

    SIXTRL_ASSERT( pz_star_squ > ZERO );

    #endif /* !defined( NDEBUG ) && !defined( __CUDACC__ ) */

    NS(Particles_set_x_value)(     p, ii, x );
    NS(Particles_set_y_value)(     p, ii, y );
    NS(Particles_set_sigma_value)( p, ii, sigma );

    NS(Particles_set_px_value)(    p, ii, px );
    NS(Particles_set_py_value)(    p, ii, py );
    NS(Particles_set_delta_value)( p, ii, delta );

    return ret;
}

SIXTRL_INLINE int NS(BeamBeam_propagate_sigma_matrix)(
        NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ptr_result,
        NS(BeamBeamPropagatedSigmasResult)* SIXTRL_RESTRICT ptr_deriv_result,
        const NS(BeamBeamSigmas) *const sigmas,
        SIXTRL_REAL_T  const s,
        SIXTRL_REAL_T  const treshold_singular,
        SIXTRL_INT64_T const handle_sigularity )
{
    int ret = 0;

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

    SIXTRL_REAL_T const dS_W = TWO *     ( sigma_12 + sigma_34 )
                             + TWO * s * ( sigma_22 + sigma_44 );

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

        SIXTRL_REAL_T const A_SQU_C_SQU_POW_3_2 =
            A_SQU_PLUS_C_SQU * SQRT_A_SQU_C_SQU;

        result.sigma_11_hat =
        result.sigma_33_hat = ONE_HALF * W;

        deriv_result.sigma_11_hat =
        deriv_result.sigma_33_hat = ONE_HALF * dS_W;

        if( A_SQU_C_SQU_POW_3_2 < treshold_singular )
        {
            SIXTRL_REAL_T const FABS_D = ( D >= ZERO ) ? D : -D;

            SIXTRL_REAL_T cos2theta = ( FABS_D > treshold_singular )
                ? ( ( B_SIGN * B ) / sqrt( B * B + FOUR * D * D ) )
                : ONE;

            result.cos_theta    = sqrt( ONE_HALF * ( ONE + cos2theta ) );
            result.sin_theta    = B_SIGN * D_SIGN * sqrt(
                ONE_HALF * ( ONE - cos2theta ) );

            deriv_result.cos_theta = ZERO;
            deriv_result.sin_theta = ZERO;
        }
        else
        {
            SIXTRL_REAL_T const SIGN_A    = ( A >= ZERO ) ? ONE : -ONE;
            SIXTRL_REAL_T const SIGN_C    = ( C >= ZERO ) ? ONE : -ONE;
            SIXTRL_REAL_T const cos2theta = ( SIGN_A * A ) / SQRT_A_SQU_C_SQU;

            SIXTRL_REAL_T const cos_theta =
                sqrt( ONE_HALF * ( ONE + cos2theta ) );

            SIXTRL_REAL_T const sin_theta =
                SIGN_A * SIGN_C * sqrt( ONE_HALF * ( ONE - cos2theta ) );


            SIXTRL_REAL_T const dS_cos2theta = SIGN_A * (
                ONE_HALF * B / SQRT_A_SQU_C_SQU -
                A * ( A * B + TWO * C * D ) / ( TWO * A_SQU_C_SQU_POW_3_2 ) );

            result.cos_theta = cos_theta;
            result.sin_theta = sin_theta;

            deriv_result.cos_theta = dS_cos2theta / ( FOUR * cos_theta );

            deriv_result.sin_theta =
                ( ( ( sin_theta > ZERO ) ? sin_theta : -sin_theta ) >
                    treshold_singular )
                    ?  -dS_cos2theta / ( FOUR * sin_theta )
                    :   D / ( A + A );

            deriv_result.sigma_11_hat += SIGN_A * SQRT_A_SQU_C_SQU;
            deriv_result.sigma_33_hat -= SIGN_A * SQRT_A_SQU_C_SQU;
        }
    }
    else
    {
        SIXTRL_REAL_T const dS_R = TWO *     ( sigma_12 - sigma_34 )
                             + TWO * s * ( sigma_22 - sigma_44 );

        SIXTRL_REAL_T const dS_sigma_13_p = sigma_14 + sigma_23
                                      + TWO * s * sigma_24;

        SIXTRL_REAL_T const dS_T = TWO * R * dS_R
                             + EIGHT * sigma_13_p * dS_sigma_13_p;

        SIXTRL_REAL_T const SQRT_T    = sqrt( T );
        SIXTRL_REAL_T const SIGN_R    = ( R >= ZERO ) ? ONE : -ONE;

        SIXTRL_REAL_T const cos2theta = SIGN_R * R / SQRT_T;
        SIXTRL_REAL_T const cos_theta = sqrt( ONE_HALF * ( ONE + cos2theta ) );

        SIXTRL_REAL_T const sin_theta =
            SIGN_R * ( ( sigma_13_p >= ZERO ) ? ONE : -ONE ) *
                sqrt( ONE_HALF * ( ONE - cos2theta ) );

        SIXTRL_REAL_T const dS_cos2theta = SIGN_R * ( dS_R / SQRT_T -
            dS_T * R / ( TWO * SQRT_T * SQRT_T * SQRT_T ) );

        SIXTRL_ASSERT( W >= SQRT_T );

        result.cos_theta    = cos_theta;
        result.sin_theta    = sin_theta;
        result.sigma_11_hat = ONE_HALF * ( W + SIGN_R * SQRT_T );
        result.sigma_33_hat = ONE_HALF * ( W - SIGN_R * SQRT_T );

        deriv_result.cos_theta = dS_cos2theta / ( FOUR * cos_theta );
        deriv_result.sin_theta =
            ( ( !handle_sigularity ) ||
              ( ( ( sin_theta >= ZERO ) ? sin_theta : -sin_theta ) <
                treshold_singular ) )
            ? -dS_cos2theta / ( FOUR * sin_theta )
            : ( sigma_14_p + sigma_23_p ) / R;

        deriv_result.sigma_11_hat =
            ONE_HALF * ( dS_W + ONE_HALF * SIGN_R * dS_T / SQRT_T  );

        deriv_result.sigma_33_hat =
            ONE_HALF * ( dS_W - ONE_HALF * SIGN_R * dS_T / SQRT_T  );
    }

    SIXTRL_ASSERT( T >= ZERO );
    SIXTRL_ASSERT( result.sigma_11_hat >= ZERO );
    SIXTRL_ASSERT( result.sigma_33_hat >= ZERO );

    *ptr_result       = result;
    *ptr_deriv_result = deriv_result;

    return ret;
}

SIXTRL_INLINE int NS(BeamBeam_get_transverse_fields)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gx_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT gy_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const min_sigma_diff )
{
    int ret = 0;

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0;

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const C =
        ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751 *
        ( SIXTRL_REAL_T )8.854187817620e-12;

    SIXTRL_REAL_T const abs_delta_sigma = ( sigma_x >= sigma_y )
        ? ( sigma_x - sigma_y ) : ( sigma_y - sigma_x );

    SIXTRL_REAL_T Ex = ZERO;
    SIXTRL_REAL_T Ey = ZERO;

    if( abs_delta_sigma < min_sigma_diff )
    {
        SIXTRL_REAL_T const sigma =
            ( SIXTRL_REAL_T )0.5L * ( sigma_x + sigma_y );

        ret = NS(BeamBeam_get_transverse_fields_gauss_round)(
            &Ex, &Ey, x, y, sigma, ZERO, ZERO );

        if( ( ret == 0 ) && ( gx_component != 0 ) && ( gy_component != 0 ) )
        {
            SIXTRL_REAL_T const TWO_SIGMA_SQU =
                ( SIXTRL_REAL_T )2.0 * sigma * sigma;

            SIXTRL_REAL_T const INV_FACTOR =
                ( SIXTRL_REAL_T )1.0 / ( TWO_SIGMA_SQU * C );

            SIXTRL_REAL_T const x_squ = x * x;
            SIXTRL_REAL_T const y_squ = y * y;
            SIXTRL_REAL_T const r_squ = x_squ + y_squ;

            SIXTRL_REAL_T const exp_r_squ = exp( -r_squ / TWO_SIGMA_SQU );
            SIXTRL_REAL_T const two_r_squ = ( SIXTRL_REAL_T )2.0 * r_squ;

            SIXTRL_ASSERT( ( gx_component != 0 ) && ( gy_component != 0 ) );

            *gx_component = ( y * Ey - x * Ex +
                x_squ * exp_r_squ * INV_FACTOR ) / ( two_r_squ );

            *gy_component = ( x * Ex - y * Ey +
                y_squ * exp_r_squ * INV_FACTOR ) / ( two_r_squ );
        }
    }
    else
    {
        ret = NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
            &Ex, &Ey, x, y, sigma_x, sigma_y, ZERO, ZERO );

        if( ( ret == 0 ) && ( gx_component != 0 ) && ( gy_component != 0 ) )
        {
            SIXTRL_REAL_T const sigma_11   = sigma_x * sigma_x;
            SIXTRL_REAL_T const sigma_33   = sigma_y * sigma_y;

            SIXTRL_REAL_T const x_squ      = x * x;
            SIXTRL_REAL_T const y_squ      = y * y;

            SIXTRL_REAL_T const exp_r_squ  =
                exp( -x_squ / ( ( SIXTRL_REAL_T )2.0 * sigma_11 )
                     -y_squ / ( ( SIXTRL_REAL_T )2.0 * sigma_33 ) );

            SIXTRL_REAL_T const INV_FACTOR =
                ( SIXTRL_REAL_T )1. / ( ( SIXTRL_REAL_T )2. * C );

            SIXTRL_REAL_T const INV_DELTA_SIGMA = ( SIXTRL_REAL_T )1. /
                ( ( SIXTRL_REAL_T )2. * ( sigma_11 - sigma_33 ) );


            *gx_component = -( x * Ex + y * Ey +
                ( INV_FACTOR * ( sigma_y / sigma_x ) * exp_r_squ -
                    ( SIXTRL_REAL_T )1.0 ) ) * INV_DELTA_SIGMA;

            *gy_component =  ( x * Ex + y * Ey +
                ( INV_FACTOR * ( sigma_x / sigma_y ) * exp_r_squ -
                    ( SIXTRL_REAL_T )1.0 ) ) * INV_DELTA_SIGMA;
        }
    }

    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );

    *ex_component = Ex;
    *ey_component = Ey;

    return ret;
}

SIXTRL_INLINE int NS(BeamBeam_get_transverse_fields_gauss_round)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma,
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y )
{
    int ret = 0;

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const C =
        ( SIXTRL_REAL_T )2.0L *
        ( SIXTRL_REAL_T )3.1415926535897932384626433832795028841971693993751L *
        ( SIXTRL_REAL_T )8.854187817620e-12L;

    SIXTRL_REAL_T const diff_x = ( x - delta_x );
    SIXTRL_REAL_T const diff_y = ( y - delta_y );
    SIXTRL_REAL_T const r_squ  = diff_x * diff_x + diff_y * diff_y;

    SIXTRL_REAL_T const temp = ( r_squ >= ( SIXTRL_REAL_T )1e-20L )
        ? ( ( SIXTRL_REAL_T )1.0 - exp( -r_squ /
            ( ( SIXTRL_REAL_T )2.0L * sigma * sigma ) ) ) / ( C * r_squ )
        : sqrt( r_squ ) / ( C * sigma );

    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );

    *ex_component = temp * diff_x;
    *ey_component = temp * diff_y;

    return ret;
}

SIXTRL_INLINE int NS(BeamBeam_get_transverse_fields_gauss_elliptical)(
        SIXTRL_REAL_T* SIXTRL_RESTRICT ex_component,
        SIXTRL_REAL_T* SIXTRL_RESTRICT ey_component,
        SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
        SIXTRL_REAL_T const sigma_x, SIXTRL_REAL_T const sigma_y,
        SIXTRL_REAL_T const delta_x, SIXTRL_REAL_T const delta_y )
{
    int ret = 0;

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const C =
        ( SIXTRL_REAL_T )2.0L *
        ( SIXTRL_REAL_T )1.7724538509055160272981674833411451827975494561224L *
        ( SIXTRL_REAL_T )8.854187817620e-12L;

    SIXTRL_REAL_T const abs_diff_x = ( x >= delta_x )
        ? ( x - delta_x ) : ( delta_x - x );

    SIXTRL_REAL_T const abs_diff_y = ( y >= delta_y )
        ? ( y - delta_y ) : ( delta_y - y );

    SIXTRL_REAL_T const ETA_ERRFUN_SCALE_FACTOR = exp( -( SIXTRL_REAL_T )0.5L *
            ( ( abs_diff_x * abs_diff_x ) / ( sigma_x * sigma_x ) +
              ( abs_diff_y * abs_diff_y ) / ( sigma_y * sigma_y ) ) );

    SIXTRL_REAL_T const HALF_S_SQU = sigma_x * sigma_x - sigma_y * sigma_y;
    SIXTRL_REAL_T factBE = ( SIXTRL_REAL_T )1.0 / C;

    SIXTRL_REAL_T eta_be_re;
    SIXTRL_REAL_T eta_be_im;

    SIXTRL_REAL_T zeta_be_re;
    SIXTRL_REAL_T zeta_be_im;

    SIXTRL_REAL_T temp_re;
    SIXTRL_REAL_T temp_im;

    SIXTRL_ASSERT( ( ex_component != 0 ) && ( ey_component != 0 ) );

    SIXTRL_ASSERT( ( sigma_x * sigma_x ) >= ( SIXTRL_REAL_T )1e-16 );
    SIXTRL_ASSERT( ( sigma_y * sigma_y ) >= ( SIXTRL_REAL_T )1e-16 );
    SIXTRL_ASSERT( ( sigma_x > sigma_y ) || ( sigma_x < sigma_y ) );
    SIXTRL_ASSERT( ( ( SIXTRL_REAL_T )1e-16 <
        ( HALF_S_SQU >= ( SIXTRL_REAL_T )0.0 ) ? HALF_S_SQU : -HALF_S_SQU ) );

    if( sigma_x > sigma_y )
    {
        SIXTRL_REAL_T const INV_S = ( SIXTRL_REAL_T )1.0L /
            sqrt( ( SIXTRL_REAL_T )2.0 * HALF_S_SQU );

        factBE    *= INV_S;
        eta_be_re   = ( sigma_y / sigma_x * abs_diff_x ) * INV_S;
        eta_be_im   = ( sigma_x / sigma_y * abs_diff_y ) * INV_S;

        zeta_be_re  = abs_diff_x * INV_S;
        zeta_be_im  = abs_diff_y * INV_S;
    }
    else if( sigma_x < sigma_y )
    {
        SIXTRL_REAL_T const INV_S = ( SIXTRL_REAL_T )1.0L /
            sqrt( ( SIXTRL_REAL_T )2.0 * -HALF_S_SQU );

        factBE    *= INV_S;
        eta_be_re   = ( sigma_x / sigma_y * abs_diff_y ) * INV_S;
        eta_be_im   = ( sigma_y / sigma_x * abs_diff_x ) * INV_S;

        zeta_be_re  = abs_diff_y * INV_S;
        zeta_be_im  = abs_diff_x * INV_S;
    }
    else
    {
        ret = -1;
    }

    NS(Faddeeva_calculate_w)( &eta_be_re, &eta_be_im, eta_be_re, eta_be_im );
    NS(Faddeeva_calculate_w)( &zeta_be_re, &zeta_be_im, zeta_be_re, zeta_be_im );

    eta_be_re *= ETA_ERRFUN_SCALE_FACTOR;
    temp_re    = ( zeta_be_re - eta_be_re ) * factBE;

    eta_be_im *= ETA_ERRFUN_SCALE_FACTOR;
    temp_im    = ( zeta_be_im - eta_be_im ) * factBE;

    if( sigma_x > sigma_y )
    {
        *ex_component = ( x >= delta_x ) ? ( temp_im ) : ( -temp_im );
        *ey_component = ( y >= delta_y ) ? ( temp_re ) : ( -temp_re );
    }
    else if( sigma_y < sigma_x )
    {
        *ex_component = ( x >= delta_x ) ? temp_re : -temp_re;
        *ey_component = ( y >= delta_y ) ? temp_im : -temp_im;
    }

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_IMPL_BEAM_BEAM_ELEMENT_6D_H__ */

/* end: sixtracklib/common/impl/bb6d_element_api.h */
