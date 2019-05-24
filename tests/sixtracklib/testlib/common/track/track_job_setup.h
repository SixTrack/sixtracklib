#ifndef SIXTRACKLIB_TESTS_TESTLIB_COMMON_TRACK_TRACK_JOB_SETUP_C99_H__
#define SIXTRACKLIB_TESTS_TESTLIB_COMMON_TRACK_TRACK_JOB_SETUP_C99_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/track/track_job_base.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(TestTrackJob_compare_particle_set_indices_lists)(
    NS(buffer_size_t) const  lhs_length,
    NS(buffer_size_t) const* SIXTRL_RESTRICT lhs_particles_set_indices_begin,
    NS(buffer_size_t) const  rhs_length,
    NS(buffer_size_t) const* SIXTRL_RESTRICT rhs_particles_set_indices_begin );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TestTrackJob_setup_no_required_output)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT track_job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ptr_output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
bool NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT track_job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ptr_output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN
bool NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT track_job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ptr_output_buffer,
    NS(buffer_size_t) const num_beam_monitors,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const until_turn_elem_by_elem );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_TESTS_TESTLIB_COMMON_TRACK_TRACK_JOB_SETUP_C99_H__ */

/* end: tests/sixtracklib/testlib/common/track/track_job_setup.h */
