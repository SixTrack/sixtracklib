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

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
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

SIXTRL_INLINE int NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    typedef NS(particle_real_t) real_t;

    real_t const minus_dx = -( NS(XYShift_dx)( xy_shift ) );
    real_t const minus_dy = -( NS(XYShift_dy)( xy_shift ) );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, index ) ==
                  ( NS(particle_index_t) )1 );

    NS(Particles_add_to_x_value)( particles, index, minus_dx );
    NS(Particles_add_to_y_value)( particles, index, minus_dy );

    return 0;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_XY_SHIFT_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_xyshift/track.h */
