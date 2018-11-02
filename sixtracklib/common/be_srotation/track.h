#ifndef SIXTRACKL_COMMON_BE_SROTATION_TRACK_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_SROTATION_TRACK_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(SRotation);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_srotation)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(SRotation) *const SIXTRL_RESTRICT srotation );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_srotation/be_srotation.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Track_particle_srotation)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    typedef NS(particle_real_t) real_t;

    real_t const sin_z = NS(SRotation_get_sin_angle)( srotation );
    real_t const cos_z = NS(SRotation_get_cos_angle)( srotation );

    real_t const x  = NS(Particles_get_x_value)(  particles, index );
    real_t const y  = NS(Particles_get_y_value)(  particles, index );
    real_t const px = NS(Particles_get_px_value)( particles, index );
    real_t const py = NS(Particles_get_py_value)( particles, index );

    real_t const x_hat  =  cos_z * x  + sin_z * y;
    real_t const y_hat  = -sin_z * x  + cos_z * y;

    real_t const px_hat =  cos_z * px + sin_z * py;
    real_t const py_hat = -sin_z * px + cos_z * py;

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, index ) ==
                  ( NS(particle_index_t) )1 );

    NS(Particles_set_x_value)(  particles, index, x_hat );
    NS(Particles_set_y_value)(  particles, index, y_hat );

    NS(Particles_set_px_value)( particles, index, px_hat );
    NS(Particles_set_py_value)( particles, index, py_hat );

    return 0;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_SROTATION_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_srotation/track.h */
