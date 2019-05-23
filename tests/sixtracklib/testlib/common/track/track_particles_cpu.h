#ifndef SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__
#define SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TestTrackCpu_track_particles_until_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const until_turn );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__ */
/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.h */
