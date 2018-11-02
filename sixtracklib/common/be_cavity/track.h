#ifndef SIXTRACKL_COMMON_BE_CAVITY_TRACK_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_CAVITY_TRACK_C99_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Cavity);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_cavity)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Cavity) *const SIXTRL_RESTRICT cavity );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_cavity/be_cavity.h"
    #include "sixtracklib/common/constants.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Track_particle_cavity)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const index,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    typedef NS(particle_real_t) real_t;

    real_t const DEG2RAD  = SIXTRL_PI / ( real_t )180.0;
    real_t const K_FACTOR = ( ( real_t )2.0 * SIXTRL_PI ) / SIXTRL_C_LIGHT;

    real_t const   beta0  = NS(Particles_get_beta0_value)( particles, index );
    real_t const   zeta   = NS(Particles_get_zeta_value)(  particles, index );
    real_t const   chi    = NS(Particles_get_chi_value)(   particles, index );
    real_t         rvv    = NS(Particles_get_rvv_value)(   particles, index );
    real_t const   tau    = zeta / ( beta0 * rvv );

    real_t const   phase  = DEG2RAD  * NS(Cavity_get_lag)( cavity ) -
                            K_FACTOR * NS(Cavity_get_frequency)( cavity ) * tau;

    real_t const energy   = chi * sin( phase ) * NS(Cavity_get_voltage)( cavity );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, index ) ==
                   ( NS(particle_index_t) )1 );

    NS(Particles_add_to_energy_value)( particles, index, energy );

    return 0;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_CAVITY_TRACK_C99_HEADER_H__ */

/* end: sixtracklib/common/be_cavity/track.h */
