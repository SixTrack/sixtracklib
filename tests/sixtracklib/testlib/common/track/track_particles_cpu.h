#ifndef SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__
#define SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TestTrackCpu_track_particles_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const until_turn );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TestTrackCpu_track_particles_elem_by_elem_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer, 
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TestTrackCpu_track_particles_line_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer, 
    NS(buffer_size_t) const num_particle_sets, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer, 
    NS(buffer_size_t) const num_line_segments, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_begin_index, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_end_index,
    NS(particle_index_t) const until_turn,
    bool const always_finish_line );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_TRACK_TRACK_PARTICLES_CPU_H__ */
/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.h */
