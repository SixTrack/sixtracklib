#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/track_job_cpu.h"
#include "sixtracklib/common/output/output_buffer.h"

#include "sixtracklib/testlib/common/particles.h"
#include "sixtracklib/testlib/testdata/testdata_files.h"

TEST( C99_TrackJobCpuTests, MinimalExample )
{
    ::st_Buffer* in_particle_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_BEAMBEAM_PARTICLES_DUMP );

    ::st_Buffer* eb = ::st_Buffer_new_from_file(
        ::st_PATH_TO_BEAMBEAM_BEAM_ELEMENTS );

    ::st_Particles* cpy_particles = ::st_Particles_buffer_get_particles(
        in_particle_buffer, 0u );

    ::st_Buffer* pb = ::st_Buffer_new( 0 );
    ::st_Particles* particles = ::st_Particles_add_copy( pb, cpy_particles );
    ASSERT_TRUE( particles != nullptr );

    ::st_buffer_size_t const NUM_ELEM_BY_ELEM_TURNS    = 5u;
    ::st_buffer_size_t const NUM_TURNS_DUMP_EVERY_TURN = 10u;
    ::st_buffer_size_t const NUM_TURNS_TOTAL           = 50u;
    ::st_buffer_size_t const SKIP_TURNS                = 5u;

    ::st_BeamMonitor* monitor_every_turn = ::st_BeamMonitor_new( eb );

    ::st_BeamMonitor_set_start( monitor_every_turn, NUM_ELEM_BY_ELEM_TURNS );
    ::st_BeamMonitor_set_skip(  monitor_every_turn, 1 );
    ::st_BeamMonitor_set_num_stores( monitor_every_turn, NUM_TURNS_DUMP_EVERY_TURN );
    ::st_BeamMonitor_set_is_rolling( monitor_every_turn, false );
    ::st_BeamMonitor_setup_for_particles( monitor_every_turn, particles );

    ::st_BeamMonitor* monitor_skip_turns = ::st_BeamMonitor_new( eb );

    ::st_BeamMonitor_set_start( monitor_skip_turns,
        NUM_ELEM_BY_ELEM_TURNS + NUM_TURNS_DUMP_EVERY_TURN );

    ::st_BeamMonitor_set_num_stores( monitor_skip_turns, 3 );
    ::st_BeamMonitor_set_skip( monitor_skip_turns, SKIP_TURNS );
    ::st_BeamMonitor_set_is_rolling( monitor_skip_turns, true );
    ::st_BeamMonitor_setup_for_particles( monitor_skip_turns, particles );


    ::st_TrackJobCpu* track_job = ::st_TrackJobCpu_new( pb, eb,
        NUM_ELEM_BY_ELEM_TURNS, NUM_TURNS_DUMP_EVERY_TURN );

    ASSERT_TRUE( track_job != nullptr );

    ::st_buffer_size_t ii = NUM_TURNS_DUMP_EVERY_TURN;
    ++ii;

    for( ; ii < NUM_TURNS_TOTAL ; ++ii )
    {
        ASSERT_TRUE( ::st_TrackJobCpu_track( track_job, ii ) );
    }

    ::st_TrackJobCpu_collect( track_job );

    ::st_Buffer const* cmp_output_buffer =
        ::st_TrackJobCpu_get_output_buffer( track_job );

    ::st_Buffer const* particle_buffer =
        ::st_TrackJobCpu_get_particle_buffer( track_job );

    ASSERT_TRUE( cmp_output_buffer != nullptr );
    ASSERT_TRUE( particle_buffer   != nullptr );

    ASSERT_TRUE( cmp_output_buffer != nullptr );
    ASSERT_TRUE( particle_buffer == pb );
    ASSERT_TRUE( eb = ::st_TrackJobCpu_get_beam_elements_buffer( track_job ) );

    ::st_TrackJobCpu_delete( track_job );
    ::st_Buffer_delete( in_particle_buffer );
    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );
}

/* end: tests/sixtracklib/common/test_track_job_cpu_c99.cpp */
