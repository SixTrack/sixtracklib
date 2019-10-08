#ifndef SIXTRACKLIB_COMMON_BE_TRICUB_TRACK_C99_H__
#define SIXTRACKLIB_COMMON_BE_TRICUB_TRACK_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(TriCub);

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_tricub)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(TriCub) *const SIXTRL_RESTRICT tricub );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/be_tricub/be_tricub.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_tricub)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const struct NS(TriCub) *const SIXTRL_RESTRICT tricub )
{
    typedef NS(be_tricub_real_t) real_t;
    typedef NS(be_tricub_int_t)  int_t;

    SIXTRL_ASSERT(  tricub != SIXTRL_NULLPTR );
    SIXTRL_ASSERT(  NS(TriCub_get_phi_size)( tricub ) > ( int_t )0u );
    SIXTRL_ASSERT(  NS(TriCub_get_ptr_const_phi)( tricub ) != SIXTRL_NULLPTR );

    SIXTRL_ASSERT(  NS(TriCub_get_dx)( tricub ) > ( real_t )0.0 );
    SIXTRL_ASSERT(  NS(TriCub_get_dy)( tricub ) > ( real_t )0.0 );
    SIXTRL_ASSERT(  NS(TriCub_get_dz)( tricub ) > ( real_t )0.0 );

    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, ii )
        == ( NS(particle_index_t) )1 );

    real_t const x  = NS(Particles_get_x_value)( particles, ii );
    real_t const y  = NS(Particles_get_y_value)( particles, ii );
    real_t const zeta = NS(Particles_get_zeta_value)( particles, ii );

    /* ..... */

    NS(Particles_set_x_value)( particles, ii, x );
    NS(Particles_set_y_value)( particles, ii, y );
    NS(Particles_set_zeta_value)( particles, ii, zeta );

    return SIXTRL_TRACK_SUCCESS;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_TRICUB_TRACK_C99_H__ */

/*end: sixtracklib/common/be_tricub/track.h */
