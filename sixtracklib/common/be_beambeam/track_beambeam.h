#ifndef SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__
#define SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(BeamBeam4D);
struct NS(BeamBeam6D);

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_6d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT bb );

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
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam4d.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam6d.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb )
{
    SIXTRL_TRACK_RETURN ret = 0;

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;

    SIXTRL_UINT64_T const data_size = NS(BeamBeam6D_get_data_size)( bb );
    bb_data_ptr_t data = NS(BeamBeam6D_get_const_data)( bb );

    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, particle_index );
    x += ( SIXTRL_REAL_T )0.0;

    NS(Particles_set_x_value)( particles, particle_index, x );

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_6d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT bb )
{
    typedef NS(beambeam6d_real_const_ptr_t)  bb_data_ptr_t;

    SIXTRL_TRACK_RETURN ret = 0;

    SIXTRL_UINT64_T const data_size = NS(BeamBeam6D_get_data_size)( bb );
    bb_data_ptr_t data = NS(BeamBeam6D_get_const_data)( bb );

    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, particle_index );
    x += ( SIXTRL_REAL_T )0.0;

    NS(Particles_set_x_value)( particles, particle_index, x );

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* end: sixtracklib/common/be_beambeam/track_beambeam.h */
