#ifndef SIXTRACKL_COMMON_BE_MULTIPOLE_TRACK_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_MULTIPOLE_TRACK_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(MultiPole);

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(MultiPole) *const SIXTRL_RESTRICT mp );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particles_range_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_begin_idx,
    NS(particle_num_elements_t) const particle_end_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BE_ARGPTR_DEC const struct NS(MultiPole) *const SIXTRL_RESTRICT mp );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_multipole/be_multipole.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(particle_real_t)  real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR index_t const TWO  = ( index_t )2;
    SIXTRL_STATIC_VAR real_t  const ZERO = ( real_t )0.0;

    index_t const order = NS(MultiPole_get_order)( mp );
    index_t index_x = TWO * order;
    index_t index_y = index_x + ( index_t )1;

    real_t dpx = NS(MultiPole_get_bal_value)( mp, index_x );
    real_t dpy = NS(MultiPole_get_bal_value)( mp, index_y );

    real_t const x      = NS(Particles_get_x_value)( particles, ii );
    real_t const y      = NS(Particles_get_y_value)( particles, ii );
    real_t const chi    = NS(Particles_get_chi_value)( particles, ii );

    real_t const hxl    = NS(MultiPole_get_hxl)( mp );
    real_t const hyl    = NS(MultiPole_get_hyl)( mp );

    #if defined( SIXTRL_ENABLE_APERATURE_CHECK      ) && \
               ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, ii ) == ( index_t )1 );

    #endif /* SIXTRL_ENABLE_APERATURE_CHECK */

    while( index_x > 0 )
    {
        real_t const zre = dpx * x - dpy * y;
        real_t const zim = dpx * y + dpy * x;

        SIXTRL_ASSERT( index_x >= TWO );
        SIXTRL_ASSERT( index_y >= TWO );

        index_x -= TWO;
        index_y -= TWO;

        dpx = NS(MultiPole_get_bal_value)( mp, index_x ) + zre;
        dpy = NS(MultiPole_get_bal_value)( mp, index_y ) + zim;
    }

    dpx = -chi * dpx;
    dpy =  chi * dpy;

    if( ( hxl > ZERO ) || ( hyl > ZERO ) || ( hxl < ZERO ) || ( hyl < ZERO ) )
    {
        real_t const delta  = NS(Particles_get_delta_value)( particles, ii );
        real_t const length = NS(MultiPole_get_length)( mp );

        real_t const hxlx   = x * hxl;
        real_t const hyly   = y * hyl;

        NS(Particles_add_to_zeta_value)( particles, ii, chi * ( hyly - hxlx ) );

        dpx += hxl + hxl * delta;
        dpy -= hyl + hyl * delta;

        if( length > ZERO )
        {
            real_t const b1l = chi * NS(MultiPole_get_bal_value)( mp, 0 );
            real_t const a1l = chi * NS(MultiPole_get_bal_value)( mp, 1 );

            dpx -= b1l * hxlx / length;
            dpy += a1l * hyly / length;
        }
    }

    NS(Particles_add_to_px_value)( particles, ii, dpx );
    NS(Particles_add_to_py_value)( particles, ii, dpy );

    return ( SIXTRL_TRACK_RETURN )0;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_range_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_end_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BE_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    SIXTRL_ASSERT( particle_idx_stride >  ( num_elem_t )0u );
    SIXTRL_ASSERT( particle_idx        >= ( num_elem_t )0u );
    SIXTRL_ASSERT( particle_idx        <= particle_end_idx );
    SIXTRL_ASSERT( particle_end_idx <= NS(Particles_get_num_of_particles)( p ) );

    for( ; particle_idx < particle_end_idx ; particle_idx += particle_idx_stride )
    {
        ret |= NS(Track_particle_multipole)( p, particle_idx, mp );
    }

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_MULTIPOLE_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_multipole/track.h */
