#ifndef SIXTRACKL_COMMON_BE_XY_SHIFT_TRACK_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_XY_SHIFT_TRACK_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(XYShift);

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particles_range_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) particle_begin_idx,
    NS(particle_num_elements_t) const particle_end_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BE_ARGPTR_DEC const struct NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    typedef NS(particle_real_t) real_t;

    real_t const minus_dx = -( NS(XYShift_get_dx)( xy_shift ) );
    real_t const minus_dy = -( NS(XYShift_get_dy)( xy_shift ) );

    #if defined( SIXTRL_ENABLE_APERATURE_CHECK      ) && \
               ( SIXTRL_ENABLE_APERATURE_CHECK == 1 )

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, ii ) == ( index_t )1 );

    #endif /* SIXTRL_ENABLE_APERATURE_CHECK */

    NS(Particles_add_to_x_value)( particles, ii, minus_dx );
    NS(Particles_add_to_y_value)( particles, ii, minus_dy );

    return 0;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_range_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) particle_idx,
    NS(particle_num_elements_t) const particle_end_idx,
    NS(particle_num_elements_t) const particle_idx_stride,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    SIXTRL_ASSERT( particle_idx_stride >  ( num_elem_t )0u );
    SIXTRL_ASSERT( particle_idx        >= ( num_elem_t )0u );
    SIXTRL_ASSERT( particle_idx        <= particle_end_idx );
    SIXTRL_ASSERT( particle_end_idx <= NS(Particles_get_num_of_particles)( p ) );

    for( ; particle_idx < particle_end_idx ; particle_idx += particle_idx_stride )
    {
        ret |= NS(Track_particle_xy_shift)( p, particle_idx, xy_shift );
    }

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_XY_SHIFT_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_xyshift/track.h */
