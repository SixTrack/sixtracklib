#ifndef SIXTRACKL_COMMON_BE_DRIFT_TRACK_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_DRIFT_TRACK_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Drift);
struct NS(DriftExact);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_drift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_drift_exact)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(DriftExact) *const SIXTRL_RESTRICT drift );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_drift/be_drift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Track_particle_drift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    typedef NS(particle_real_t) real_t;

    real_t const rpp    = NS(Particles_get_rpp_value)( p, ii );
    real_t const xp     = NS(Particles_get_px_value )( p, ii ) * rpp;
    real_t const yp     = NS(Particles_get_py_value )( p, ii ) * rpp;
    real_t const length = NS(Drift_get_length)( drift );
    real_t const dzeta  = NS(Particles_get_rvv_value)( p, ii ) -
                          ( ( real_t )1 + ( xp*xp + yp*yp ) / ( real_t )2 );

    #if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
               ( SIXTRL_ENABLE_APERTURE_CHECK == 1 )

    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0;

    real_t const x = NS(Particles_get_x_value)( p, ii ) + length * xp;
    real_t const y = NS(Particles_get_y_value)( p, ii ) + length * yp;

    real_t const sign_x = ( real_t )( ( ZERO < x ) - ( real_t )( x < ZERO ) );
    real_t const sign_y = ( real_t )( ( ZERO < y ) - ( real_t )( y < ZERO ) );

    index_t const new_state =
        ( index_t )( ( ( sign_x * x ) < SIXTRL_APERTURE_X_LIMIT ) &
                     ( ( sign_y * y ) < SIXTRL_APERTURE_Y_LIMIT ) );

    int const ret = ( int )new_state - 1;

    SIXTRL_ASSERT( NS(Particles_get_state_value)( p, ii ) == ( index_t )1 );

    NS(Particles_set_x_value)( p, ii, x );
    NS(Particles_set_y_value)( p, ii, y );
    NS(Particles_set_state_value)( p, ii, new_state );

    #else /* SIXTRL_ENABLE_APERTURE_CHECK */

    int const ret = 0;

    NS(Particles_add_to_x_value)( p, ii, xp * length );
    NS(Particles_add_to_y_value)( p, ii, yp * length );

    #endif /* SIXTRL_ENABLE_APERTURE_CHECK */

    SIXTRL_ASSERT( NS(Particles_get_beta0_value)( p, ii ) > ( real_t )0 );

    NS(Particles_add_to_s_value)( p, ii, length );
    NS(Particles_add_to_zeta_value)( p, ii, length * dzeta );

    return ret;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Track_particle_drift_exact)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    typedef NS(particle_real_t) real_t;

    real_t const length = NS(DriftExact_get_length)( drift );
    real_t const px     = NS(Particles_get_px_value)( p, ii );
    real_t const py     = NS(Particles_get_py_value)( p, ii );

    real_t const opd    = NS(Particles_get_delta_value)( p, ii ) + ( real_t )1;
    real_t const lpzi   = length / sqrt( opd * opd - ( px * px + py * py ) );

    real_t const dzeta  = NS(Particles_get_rvv_value)( p, ii ) * length
                        - opd * lpzi;

    #if defined( SIXTRL_ENABLE_APERTURE_CHECK      ) && \
               ( SIXTRL_ENABLE_APERTURE_CHECK == 1 )

    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0;

    real_t const x = NS(Particles_get_x_value)( p, ii ) + px * lpzi;
    real_t const y = NS(Particles_get_y_value)( p, ii ) + py * lpzi;

    real_t const sign_x = ( real_t )( ( ZERO < x ) - ( real_t )( x < ZERO ) );
    real_t const sign_y = ( real_t )( ( ZERO < y ) - ( real_t )( y < ZERO ) );

    index_t const new_state =
        ( index_t )( ( ( sign_x * x ) < SIXTRL_APERTURE_X_LIMIT ) &
                     ( ( sign_y * y ) < SIXTRL_APERTURE_Y_LIMIT ) );

    int const ret = ( int )new_state - 1;

    SIXTRL_ASSERT( NS(Particles_get_state_value)( p, ii ) == ( index_t )1 );

    NS(Particles_set_x_value)( p, ii, x );
    NS(Particles_set_y_value)( p, ii, y );
    NS(Particles_set_state_value)( p, ii, new_state );

    #else /* SIXTRL_ENABLE_APERTURE_CHECK */

    int const ret = 0;

    NS(Particles_add_to_x_value)( p, ii, px * lpzi );
    NS(Particles_add_to_y_value)( p, ii, py * lpzi );

    #endif /* SIXTRL_ENABLE_APERTURE_CHECK */

    SIXTRL_ASSERT( NS(Particles_get_beta0_value)( p, ii ) > ( real_t )0 );
    SIXTRL_ASSERT( ( opd * opd ) >   ( px * px + py * py ) );
    SIXTRL_ASSERT( sqrt( opd * opd - ( px * px + py * py ) ) > ( real_t )0 );

    NS(Particles_add_to_s_value)(    p, ii, length );
    NS(Particles_add_to_zeta_value)( p, ii, dzeta );

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_DRIFT_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_drift/track.h */
