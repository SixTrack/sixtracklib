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

SIXTRL_HOST_FN NS(Buffer)* NS(TrackTestdata_extract_initial_particles_buffer)(
    const char path_to_file[] );

SIXTRL_HOST_FN NS(Buffer)* NS(TrackTestdata_extract_result_particles_buffer)(
    const char path_to_file[] );

SIXTRL_HOST_FN NS(Buffer)* NS(TrackTestdata_extract_beam_elements_buffer)(
    const char path_to_file[] );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TRACK_TESTDATA_HEADER_H__ */

/* end: tests/sixtracklib/testlib/testdata/track_testdata.h */
