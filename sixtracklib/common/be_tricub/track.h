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

SIXTRL_STATIC SIXTRL_FN void NS(construct_b_vector)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(TriCubData) *const SIXTRL_RESTRICT tricub_data,
    SIXTRL_ARGPTR_DEC NS(be_tricub_real_t)* SIXTRL_RESTRICT b_vector);

SIXTRL_STATIC SIXTRL_FN void NS(construct_coefs)(
    SIXTRL_ARGPTR_DEC NS(be_tricub_real_t) const* SIXTRL_RESTRICT b_vector,
    SIXTRL_ARGPTR_DEC NS(be_tricub_real_t)* coefs);

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
    
    const real_t fx = (x - NS(TriCubData_x0) / NS(TriCubData_dx;
    const real_t fy = (y - NS(TriCubData_y0) / NS(TriCubData_dy;
    const real_t fz = (z - NS(TriCubData_z0) / NS(TriCubData_dz;

    const real_t ixf = floor(fx);
    const real_t iyf = floor(fy);
    const real_t izf = floor(fz);

    const int_t ix = (int_t)ixf;
    const int_t iy = (int_t)iyf;
    const int_t iz = (int_t)izf;

    const real_t xn = fx - ixf;
    const real_t yn = fy - iyf;
    const real_t zn = fz - izf;

    int_t inside_box = 1;
    if      ( ix < 0 || ix > NS(TriCubData_nx)( tricub_data ) - 2 )
        inside_box = 0;
    else if ( iy < 0 || iy > NS(TriCubData_ny)( tricub_data ) - 2 )
        inside_box = 0;
    else if ( iz < 0 || iz > NS(TriCubData_nz)( tricub_data ) - 2 )
        inside_box = 0;

    real_t b_vector[64];
    NS(construct_b_vector)(tricub_data, b_vector);
    real_t coefs[64];
    NS(construct_coefs)(b_vector, coefs);

    real_t x_powers[4];
    real_t y_powers[4];
    real_t z_powers[4];

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
