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

    SIXTRL_BUFFER_DATAPTR_DEC NS(TriCubData) const* tricub_data =
        NS(TriCub_const_data)( tricub );

    SIXTRL_BUFFER_DATAPTR_DEC real_t const* lookup_table_begin =
        NS(TriCubData_const_table_begin)( tricub_data );

    int_t const lookup_table_size = NS(TriCubData_table_size)( tricub_data );

    /* How to access the data members of the NS(TriCub) beam element */
    /*
    real_t const x_closed_orbit = NS(TriCub_x)( tricub );
    real_t const y_closed_orbit = NS(TriCub_y)( tricub );
    real_t const z_closed_orbit = NS(TriCub_z)( tricub );
    real_t const length         = NS(TriCub_length)( tricub );
    */

    /* How to access the data members of NS(TriCubData) element */
    /*
     real_t const x0 = NS(TriCubData_x0)( tricub_data );
     real_t const dx = NS(TriCubData_dx)( tricub_data );
     int_t  const nx = NS(TriCubData_nx)( tricub_data );

     real_t const y0 = NS(TriCubData_x0)( tricub_data );
     real_t const dy = NS(TriCubData_dx)( tricub_data );
     int_t  const ny = NS(TriCubData_nx)( tricub_data );

     real_t const z0 = NS(TriCubData_x0)( tricub_data );
     real_t const dz = NS(TriCubData_dx)( tricub_data );
     int_t  const nz = NS(TriCubData_nx)( tricub_data );
    */

    /* how to mark a particle in a particle set as lost */
    /*
    NS(Particles_mark_as_lost_value)( particles, ii );
    */

    /* ..... */

    /* How to update the particles state at the end of *
     * applying the tracking map */
    /*
    NS(Particles_set_x_value)( particles, ii, new_x_value );
    NS(Particles_set_y_value)( particles, ii, new_y_value );
    NS(Particles_set_zeta_value)( particles, ii, new_zeta_value );
    */

    ( void )particles;
    ( void )ii;
    ( void )lookup_table_begin;
    ( void )lookup_table_size;

    return SIXTRL_TRACK_SUCCESS;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_TRICUB_TRACK_C99_H__ */

/*end: sixtracklib/common/be_tricub/track.h */
