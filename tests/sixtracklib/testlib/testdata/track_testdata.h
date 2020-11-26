#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TRACK_TESTDATA_HEADER_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TRACK_TESTDATA_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_initial_particles_buffer)( const char path_to_file[] );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_result_particles_buffer)(
    const char path_to_file[] );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_beam_elements_buffer)( const char path_to_file[] );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackTestdata_generate_fodo_lattice)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned int const num_turns );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackTestdata_generate_particle_distr_x)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned int const NUM_PARTICLES, double const p0c,
    double const min_x, double const max_x, double const mass0,
    double const q0, double const chi, double const charge_ratio );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TRACK_TESTDATA_HEADER_H__ */
