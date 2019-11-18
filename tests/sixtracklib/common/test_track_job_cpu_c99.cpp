#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/track_job_cpu.h"


namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool test1_CreateTrackJobNoOutputDelete(
            const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer );

        bool test2_CreateTrackJobElemByElemOutputDelete(
            const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
            ::NS(buffer_size_t) const target_num_elem_by_elem_turns );

        bool test3_CreateTrackJobFullOutput(
            const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
            ::NS(buffer_size_t) const num_beam_monitors,
            ::NS(buffer_size_t) const target_num_output_turns,
            ::NS(buffer_size_t) const target_num_elem_by_elem_turns );
    }
}

TEST( C99_TrackJobCpuTests, CreateTrackJobNoOutputDelete )
{
    using track_job_t   = ::st_TrackJobCpu;
    using size_t        = ::st_buffer_size_t;
    using buffer_t      = ::st_Buffer;
    using particles_t   = ::st_Particles;

    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t* particles = ::NS(Particles_add_copy)( pb,
        ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    ( void )particles;

    SIXTRL_ASSERT( eb != nullptr );
    SIXTRL_ASSERT( pb != nullptr );
    SIXTRL_ASSERT( particles          != nullptr );
    SIXTRL_ASSERT( in_particle_buffer != nullptr );
    SIXTRL_ASSERT( my_output_buffer   != nullptr );

    /* ===================================================================== *
     * First set of tests:
     * No Beam Monitors
     * No Elem by Elem config
     * --------------------------------------------------------------------- */

    track_job_t* job = ::NS(TrackJobCpu_create)();
    ASSERT_TRUE( job != nullptr );

    ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) == ::NS(TRACK_JOB_CPU_ID) );

    ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CPU_STR) ) == 0 );

    bool success = ::NS(TrackJobCpu_reset_with_output)(
        job, pb, eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job, pb, eb, nullptr ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_with_output)( pb, eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) == ::NS(TRACK_JOB_CPU_ID) );

    ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CPU_STR) ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job, pb, eb, nullptr ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) == ::NS(TRACK_JOB_CPU_ID) );

    ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CPU_STR) ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job, pb, eb, nullptr ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const wrong_particle_sets[] =
    {
        size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
    };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 3 },
        &wrong_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) == ::NS(TRACK_JOB_CPU_ID) );

    ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CPU_STR) ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job, pb, eb, nullptr ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, my_output_buffer, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) == ::NS(TRACK_JOB_CPU_ID) );

    ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CPU_STR) ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job, pb, eb, my_output_buffer ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* ===================================================================== */

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( my_output_buffer );
}

TEST( C99_TrackJobCpuTests, CreateTrackJobElemByElemOutputDelete )
{
    using track_job_t         = ::st_TrackJobCpu;
    using size_t              = ::st_buffer_size_t;
    using buffer_t            = ::st_Buffer;
    using particles_t         = ::st_Particles;

    namespace st_test         = SIXTRL_CXX_NAMESPACE::tests;

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );

    buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( pb != nullptr );
    SIXTRL_ASSERT( eb != nullptr );
    SIXTRL_ASSERT( in_particle_buffer != nullptr );

    buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t* particles = ::NS(Particles_add_copy)( pb,
        ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( my_output_buffer != nullptr );

    size_t const NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );

    size_t const NUM_PARTICLES =
        ::NS(Particles_get_num_of_particles)( particles );

    size_t const DUMP_ELEM_BY_ELEM_TURNS = size_t{  5u };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    /* ===================================================================== *
     * Second set of tests:
     * No Beam Monitors
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t* job = ::NS(TrackJobCpu_create)();
    ASSERT_TRUE( job != nullptr );

    bool success = ::NS(TrackJobCpu_reset_with_output)(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_with_output)(
        pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr,
        DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const wrong_particle_sets[] =
    {
        size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
    };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 3 },
        &wrong_particle_sets[ 0 ], eb, nullptr,
        DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, my_output_buffer,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job, pb, eb, my_output_buffer, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* ===================================================================== */

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( my_output_buffer );
}

TEST( C99_TrackJobCpuTests, CreateTrackJobBeamMonitorOutputDelete )
{
    using track_job_t  = ::NS(TrackJobCpu);
    using size_t       = ::NS(buffer_size_t);
    using buffer_t     = ::NS(Buffer);
    using particles_t  = ::NS(Particles);
    using be_monitor_t = ::NS(BeamMonitor);

    namespace st_test  = SIXTRL_CXX_NAMESPACE::tests;

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );

    buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( pb != nullptr );
    SIXTRL_ASSERT( eb != nullptr );
    SIXTRL_ASSERT( in_particle_buffer != nullptr );

    buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t* particles = ::NS(Particles_add_copy)( pb,
        ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( my_output_buffer != nullptr );

    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    size_t NUM_PARTICLES = ::NS(Particles_get_num_of_particles)( particles );
    size_t NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );

    be_monitor_t* turn_by_turn_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)( turn_by_turn_monitor, false );
    ::NS(BeamMonitor_set_start)( turn_by_turn_monitor, size_t{ 0 } );
    ::NS(BeamMonitor_set_num_stores)(
        turn_by_turn_monitor, NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( eot_monitor != nullptr );

    ::NS(BeamMonitor_set_skip)( eot_monitor, true );
    ::NS(BeamMonitor_set_is_rolling)( eot_monitor, SKIP_TURNS );
    ::NS(BeamMonitor_set_start)( eot_monitor, NUM_TURN_BY_TURN_TURNS );
    ::NS(BeamMonitor_set_num_stores)( eot_monitor,
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    ASSERT_TRUE( NUM_PARTICLES == static_cast< size_t >(
        ::NS(Particles_get_num_of_particles)( particles ) ) );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( ::NS(Buffer_get_num_of_objects)( eb ) ) );

    /* ===================================================================== *
     * Third set of tests:
     * Two Beam Monitors at end of lattice
     * No Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t* job = ::NS(TrackJobCpu_create)();
    ASSERT_TRUE( job != nullptr );

    bool success = ::NS(TrackJobCpu_reset)( job, pb, eb, nullptr );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new)( pb, eb );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const wrong_particle_sets[] =
    {
        size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
    };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 3 },
        &wrong_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, my_output_buffer, size_t{ 0 }, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, my_output_buffer, NUM_BEAM_MONITORS,
        NUM_TURNS, size_t{ 0 } ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* ===================================================================== */

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( my_output_buffer );
}

TEST( C99_TrackJobCpuTests, CreateTrackJobFullDelete )
{
    using track_job_t  = ::NS(TrackJobCpu);
    using size_t       = ::NS(buffer_size_t);
    using buffer_t     = ::NS(Buffer);
    using particles_t  = ::NS(Particles);
    using be_monitor_t = ::NS(BeamMonitor);

    namespace st_test  = SIXTRL_CXX_NAMESPACE::tests;

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );

    buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( pb != nullptr );
    SIXTRL_ASSERT( eb != nullptr );
    SIXTRL_ASSERT( in_particle_buffer != nullptr );

    buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t* particles = ::NS(Particles_add_copy)( pb,
        ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( my_output_buffer != nullptr );

    size_t const DUMP_ELEM_BY_ELEM_TURNS  = size_t{ 5 };
    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    size_t NUM_PARTICLES = ::NS(Particles_get_num_of_particles)( particles );
    size_t NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );

    be_monitor_t* turn_by_turn_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)( turn_by_turn_monitor, false );

    ::NS(BeamMonitor_set_start)(
        turn_by_turn_monitor, DUMP_ELEM_BY_ELEM_TURNS );

    ::NS(BeamMonitor_set_num_stores)(
        turn_by_turn_monitor, NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( eot_monitor != nullptr );

    ::NS(BeamMonitor_set_skip)( eot_monitor, true );

    ::NS(BeamMonitor_set_is_rolling)( eot_monitor, SKIP_TURNS );

    ::NS(BeamMonitor_set_start)( eot_monitor,
            DUMP_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS );

    ::NS(BeamMonitor_set_num_stores)( eot_monitor,
        ( NUM_TURNS - ( DUMP_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS ) ) /
            SKIP_TURNS );

    ASSERT_TRUE( NUM_PARTICLES == static_cast< size_t >(
        ::NS(Particles_get_num_of_particles)( particles ) ) );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( ::NS(Buffer_get_num_of_objects)( eb ) ) );

    /* ===================================================================== *
     * Fourth set of tests:
     * Two Beam Monitors at end of lattice
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t* job = ::NS(TrackJobCpu_create)();
    ASSERT_TRUE( job != nullptr );

    bool success = ::NS(TrackJobCpu_reset_with_output)(
        job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
            DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_with_output)(
        pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
            DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr,
        DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job, pb, eb, nullptr,
        NUM_BEAM_MONITORS, NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    size_t const wrong_particle_sets[] =
    {
        size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
    };

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 3 },
        &wrong_particle_sets[ 0 ], eb, nullptr,
        DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job, pb, eb, nullptr,
        NUM_BEAM_MONITORS, NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* --------------------------------------------------------------------- */

    job = ::NS(TrackJobCpu_new_detailed)( pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, my_output_buffer,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job, pb, eb, my_output_buffer, NUM_BEAM_MONITORS,
        NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

    ::NS(TrackJobCpu_delete)( job );
    job = nullptr;

    /* ===================================================================== */

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( my_output_buffer );
}

TEST( C99_TrackJobCpuTests, CreateTrackJobTrackLineCompare )
{
    using buffer_t    = ::NS(Buffer)*;
    using particles_t = ::NS(Particles)*;
    using buf_size_t  = ::NS(buffer_size_t);
    using track_job_t = ::NS(TrackJobCpu);

    buffer_t pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t track_pb     = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    particles_t particles = ::NS(Particles_add_copy)( track_pb,
        ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

    particles_t cmp_particles = ::NS(Particles_add_copy)( cmp_track_pb,
        ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( cmp_particles != nullptr );

    buf_size_t const until_turn = 10;
    int status = ::NS(Track_all_particles_until_turn)(
        cmp_particles, eb, until_turn );

    SIXTRL_ASSERT( status == 0 );

    buf_size_t const num_beam_elements = ::NS(Buffer_get_num_of_objects)( eb );
    buf_size_t const num_lattice_parts = buf_size_t{ 10 };
    buf_size_t const num_elem_per_part = num_beam_elements / num_lattice_parts;

    track_job_t* job = ::NS(TrackJobCpu_new)( track_pb, eb );
    ASSERT_TRUE( job != nullptr );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < until_turn ; ++ii )
    {
        for( buf_size_t jj = buf_size_t{ 0 } ; jj < num_lattice_parts ; ++jj )
        {
            bool const is_last_in_turn = ( jj == ( num_lattice_parts - 1 ) );
            buf_size_t const begin_idx =  jj * num_elem_per_part;
            buf_size_t const end_idx   = ( !is_last_in_turn ) ?
                begin_idx + num_elem_per_part : num_beam_elements;

            status = ::NS(TrackJobCpu_track_line)(
                job, begin_idx, end_idx, is_last_in_turn );
            ASSERT_TRUE( status == 0 );
        }
    }

    double const ABS_DIFF = double{ 2e-14 };

    if( ( 0 != ::NS(Particles_compare_values)( cmp_particles, particles ) ) &&
        ( 0 != ::NS(Particles_compare_values_with_treshold)(
            cmp_particles, particles, ABS_DIFF ) ) )
    {
        buffer_t diff_buffer  = ::NS(Buffer_new)( buf_size_t{ 0 } );
        SIXTRL_ASSERT( diff_buffer != nullptr );

        particles_t diff = ::NS(Particles_new)( diff_buffer,
            NS(Particles_get_num_of_particles)( cmp_particles ) );

        ::NS(Particles_calculate_difference)( particles, cmp_particles, diff );

        printf( "particles: \r\n" );
        ::NS(Particles_print_out)( particles );

        printf( "cmp_particles: \r\n" );
        ::NS(Particles_print_out)( cmp_particles );

        printf( "diff: \r\n" );
        ::NS(Particles_print_out)( diff );

        ::NS(Buffer_delete)( diff_buffer );
    }

    ASSERT_TRUE(
        ( 0 == ::NS(Particles_compare_values)( cmp_particles, particles ) ) ||
        ( 0 == ::NS(Particles_compare_values_with_treshold)(
            cmp_particles, particles, ABS_DIFF ) ) );

    ::NS(TrackJobCpu_delete)( job );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( track_pb );
    ::NS(Buffer_delete)( cmp_track_pb );

}


TEST( C99_TrackJobCpuTests, TrackParticles )
{
    using track_job_t  = ::NS(TrackJobCpu);
    using size_t       = ::NS(buffer_size_t);
    using buffer_t     = ::NS(Buffer);
    using particles_t  = ::NS(Particles);
    using part_index_t = ::NS(particle_index_t);

    namespace st_test  = SIXTRL_CXX_NAMESPACE::tests;

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );

    buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( pb != nullptr );
    SIXTRL_ASSERT( eb != nullptr );
    SIXTRL_ASSERT( in_particle_buffer != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)( pb,
        ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );

    buffer_t* cmp_pb            = ::NS(Buffer_new)( size_t{ 0 } );
    buffer_t* cmp_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t* cmp_particles = ::NS(Particles_add_copy)(
        cmp_pb, ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    size_t const DUMP_ELEM_BY_ELEM_TURNS = size_t{   5 };
    size_t const NUM_TURNS               = size_t{ 100 };
    size_t const SKIP_TURNS              = size_t{  10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{  10 };
    size_t const NUM_BEAM_MONITORS       = size_t{   2 };

    size_t NUM_PARTICLES = ::NS(Particles_get_num_of_particles)( particles );
    size_t NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );

    int ret = ::NS(BeamMonitor_insert_end_of_turn_monitors)(
        eb, DUMP_ELEM_BY_ELEM_TURNS, NUM_TURN_BY_TURN_TURNS,
            NUM_TURNS, SKIP_TURNS, NS(Buffer_get_objects_end)( eb ) );

    SIXTRL_ASSERT( ret == 0 );

    ASSERT_TRUE( NUM_PARTICLES == static_cast< size_t >(
        ::NS(Particles_get_num_of_particles)( particles ) ) );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( ::NS(Buffer_get_num_of_objects)( eb ) ) );

    /* -------------------------------------------------------------------- */
    /* create cmp particle and output data to verify track job performance  */

    size_t elem_by_elem_offset = size_t{ 0 };
    size_t beam_monitor_offset = size_t{ 0 };
    size_t num_elem_by_elem_objs = size_t{ 0 };

    part_index_t const start_elem = part_index_t{ 0u };
    part_index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
                 min_turn_id, max_turn_id;

    ret = ::NS(OutputBuffer_get_min_max_attributes)( cmp_particles, eb,
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
        &min_turn_id, &max_turn_id, &num_elem_by_elem_objs, start_elem );

    part_index_t const max_elem_by_elem_turn_id = (
        ( DUMP_ELEM_BY_ELEM_TURNS > static_cast< size_t >( min_turn_id ) ) &&
        ( min_turn_id >= part_index_t{ 0 } ) )
        ? ( DUMP_ELEM_BY_ELEM_TURNS - 1 ) : min_turn_id;

    ::NS(ElemByElemConfig) config;
    ::NS(ElemByElemConfig_preset)( &config );

    if( ( NS(TRACK_SUCCESS) == ret ) &&
        ( static_cast< size_t >( min_turn_id ) < DUMP_ELEM_BY_ELEM_TURNS ) &&
        ( num_elem_by_elem_objs > size_t{ 0 } ) )
    {
        ret = ::NS(ElemByElemConfig_init_detailed)( &config,
            NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES), min_part_id,
                max_part_id, min_elem_id, max_elem_id,
                    min_turn_id, max_elem_by_elem_turn_id, true );
    }
    else if( ret == ::NS(TRACK_SUCCESS) )
    {
        ret = ::NS(TRACK_STATUS_GENERAL_FAILURE);
    }

    ret = ::NS(OutputBuffer_prepare)( eb, cmp_output_buffer, cmp_particles,
            DUMP_ELEM_BY_ELEM_TURNS, &elem_by_elem_offset,
                &beam_monitor_offset, &min_turn_id );

    if( ret == ::NS(TRACK_SUCCESS) )
    {
        ret = ::NS(BeamMonitor_assign_output_buffer_from_offset)(
            eb, cmp_output_buffer, min_turn_id, beam_monitor_offset );
    }

    if( ret == NS(TRACK_SUCCESS) )
    {
        ::NS(ElemByElemConfig_set_output_store_address)(
            &config, ( uintptr_t )::NS(Particles_buffer_get_const_particles)(
                cmp_output_buffer, elem_by_elem_offset ) );
    }

    ret = ::NS(Track_all_particles_element_by_element_until_turn)(
        cmp_particles, &config, eb, DUMP_ELEM_BY_ELEM_TURNS );

    SIXTRL_ASSERT( ret == ::NS(TRACK_SUCCESS) );

    ret = ::NS(Track_all_particles_until_turn)( cmp_particles, eb, NUM_TURNS );

    SIXTRL_ASSERT( ret == 0 );

    ::NS(BeamMonitor_clear_all)( eb );

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    track_job_t* job = ::NS(TrackJobCpu_new_with_output)(
        pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_has_output_buffer)( job ) );
    ASSERT_TRUE( ::NS(TrackJob_owns_output_buffer)( job ) );
    ASSERT_TRUE( ::NS(TrackJob_get_const_output_buffer)( job ) != nullptr );
    ASSERT_TRUE( ::NS(TrackJob_has_beam_monitor_output)( job ) );
    ASSERT_TRUE( ::NS(TrackJob_get_num_beam_monitors)( job ) ==
                 NUM_BEAM_MONITORS );

    ASSERT_TRUE( ::NS(TrackJob_has_elem_by_elem_output)( job ) );
    ASSERT_TRUE( ::NS(TrackJob_has_elem_by_elem_config)( job ) );
    ASSERT_TRUE( ::NS(TrackJob_get_elem_by_elem_config)( job ) != nullptr );

    ret = ::NS(TrackJobCpu_track_elem_by_elem)( job, DUMP_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( ret == 0 );

    ret = ::NS(TrackJobCpu_track_until_turn)( job, NUM_TURNS );
    ASSERT_TRUE( ret == 0 );

    ::NS(TrackJobCpu_collect)( job );

    /* --------------------------------------------------------------------- */
    /* compare */

    buffer_t const* ptr_output_buffer =
        ::NS(TrackJob_get_const_output_buffer)( job );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( cmp_output_buffer ) );

    double const ABS_ERR = double{ 1e-14 };

    if( ::NS(Particles_buffers_compare_values)(
            ptr_output_buffer, cmp_output_buffer ) != 0 )
    {
        if( ::NS(Particles_buffers_compare_values_with_treshold)(
            ptr_output_buffer, cmp_output_buffer, ABS_ERR ) != 0 )
        {
            size_t const nn =
                ::NS(Buffer_get_num_of_objects)( cmp_output_buffer );

            buffer_t* diff_buffer = ::NS(Buffer_new)( size_t{ 0 } );

            for( size_t ii =  size_t{ 0 } ; ii < nn ; ++ii )
            {
                particles_t const* cmp =
                    ::NS(Particles_buffer_get_const_particles)(
                        cmp_output_buffer, ii );

                particles_t const* trk_particles =
                    ::NS(Particles_buffer_get_const_particles)(
                        ptr_output_buffer, ii );

                ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp ) ==
                    ::NS(Particles_get_num_of_particles)( trk_particles ) );

                ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp ) > 0 );

                if( 0 != ::NS(Particles_compare_values_with_treshold)(
                        cmp, trk_particles, ABS_ERR ) )
                {
                    particles_t* diff = ::NS(Particles_new)( diff_buffer,
                        ::NS(Particles_get_num_of_particles)( cmp ) );

                    SIXTRL_ASSERT( diff != nullptr );

                    ::NS(Particles_calculate_difference)(
                        cmp, trk_particles, diff );

                    size_t const mm = ::NS(Particles_get_num_of_particles)(
                        cmp );

                    std::cout << "ii = " << ii << std::endl;

                    for( size_t ll = size_t{ 0 } ; ll < mm ; ++ll )
                    {
                        std::cout << "trk_particles:\r\n";
                        ::NS(Particles_print_out_single)( trk_particles, ll );

                        std::cout << "cmp:\r\n";
                        ::NS(Particles_print_out_single)( cmp, ll );

                        std::cout << "diff:\r\n";
                        ::NS(Particles_print_out_single)( diff, ll );
                        std::cout << "\r\n"
                                  << "---------------------------------------"
                                  << "---------------------------------------"
                                  << "\r\n";
                    }

                    std::cout << std::endl;
                }
            }

            ::NS(Buffer_delete)( diff_buffer );
            diff_buffer = nullptr;
        }
    }

    ASSERT_TRUE( ( ::NS(Particles_buffers_compare_values)(
                        ptr_output_buffer, cmp_output_buffer ) == 0 ) ||
                 ( ::NS(Particles_buffers_compare_values_with_treshold)(
                        ptr_output_buffer, cmp_output_buffer, ABS_ERR ) == 0 ) );

    /* --------------------------------------------------------------------- */

    ::NS(TrackJobCpu_delete)( job );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( cmp_output_buffer );
    ::NS(Buffer_delete)( cmp_pb );
}


namespace SIXTRL_CXX_NAMESPACE
{
namespace tests
{
    bool test1_CreateTrackJobNoOutputDelete(
         const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
         const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
         const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
         const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer )
    {
        using size_t = ::NS(buffer_size_t);

        bool success = (
            ( job != nullptr ) &&
            ( ( ( ext_output_buffer == nullptr ) &&
                ( !::NS(TrackJob_has_output_buffer)(  job ) ) &&
                ( !::NS(TrackJob_owns_output_buffer)( job ) ) ) ||
              ( ( ext_output_buffer != nullptr ) &&
                (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                ( !::NS(TrackJob_owns_output_buffer)( job ) ) ) ) &&
            ( !::NS(TrackJob_has_beam_monitor_output)( job ) ) &&
            (  ::NS(TrackJob_get_num_beam_monitors)( job ) == size_t{ 0 } ) );

        if( success )
        {
            success = (
                ( ::NS(TrackJob_get_beam_monitor_indices_begin)(
                    job ) == nullptr ) &&
                ( ::NS(TrackJob_get_beam_monitor_indices_end)(
                    job ) == nullptr ) );
        }

        if( success )
        {
            success = (
                ( ::NS(TrackJob_get_num_particle_sets)(
                    job ) == size_t{ 1 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_begin)(
                    job ) != nullptr ) &&
                ( ::NS(TrackJob_get_particle_set_indices_end)(
                    job ) != nullptr ) &&
                ( *( ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) ==
                    size_t{ 0 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_end)( job ) !=
                  ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) &&
                ( ::NS(TrackJob_get_particle_set_index)( job, size_t{ 0 } ) ==
                    size_t{ 0 } ) );
        }

        if( success )
        {
            success = (
                ( !::NS(TrackJob_has_elem_by_elem_output)( job ) ) &&
                ( !::NS(TrackJob_has_elem_by_elem_config)( job ) ) &&
                (  ::NS(TrackJob_get_elem_by_elem_config)(
                    job ) == nullptr ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = (
                    (  ::NS(TrackJob_get_const_output_buffer)(
                        job ) == ext_output_buffer ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
            else
            {
                success = (
                    ( ::NS(TrackJob_get_const_output_buffer)(
                        job ) == nullptr ) &&
                    ( !::NS(TrackJob_has_output_buffer)( job ) ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
        }

        if( success )
        {
            success = ( ( ::NS(TrackJob_get_const_particles_buffer)(
                            job ) == particles_buffer ) &&
                        ( ::NS(TrackJob_get_const_beam_elements_buffer)(
                            job ) == beam_elements_buffer ) );
        }

        return success;
    }

    bool test2_CreateTrackJobElemByElemOutputDelete(
            const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
            ::NS(buffer_size_t) const target_num_elem_by_elem_turns )
    {
        bool success = false;

        using size_t              = ::st_buffer_size_t;
        using buffer_t            = ::st_Buffer;
        using particles_t         = ::st_Particles;
        using elem_by_elem_conf_t = ::NS(ElemByElemConfig);

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        SIXTRL_ASSERT( target_num_elem_by_elem_turns > size_t{ 0 } );
        SIXTRL_ASSERT( particles_buffer != nullptr );
        SIXTRL_ASSERT( beam_elements_buffer != nullptr );

        particles_t const* particles =
        ::NS(Particles_buffer_get_const_particles)(
                particles_buffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS =
            ::NS(Buffer_get_num_of_objects)( beam_elements_buffer );

        size_t const NUM_PARTICLES =
            ::NS(Particles_get_num_of_particles)( particles );

        success = ( ( job != nullptr ) &&
            ( NUM_BEAM_ELEMENTS > size_t{ 0 } ) &&
            ( NUM_PARTICLES > size_t{ 0 } ) &&
            (  ::NS(TrackJob_has_output_buffer)(  job ) ) &&
            ( !::NS(TrackJob_has_beam_monitor_output)( job ) ) &&
            (  ::NS(TrackJob_get_num_beam_monitors)( job ) == size_t{ 0 } ) &&
            (  ::NS(TrackJob_get_beam_monitor_indices_begin)(
                job ) == nullptr ) &&
            (  ::NS(TrackJob_get_beam_monitor_indices_end)(
                job ) == nullptr ) );

        if( success )
        {
            success = (
                ( ::NS(TrackJob_get_num_particle_sets)(
                    job ) == size_t{ 1 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_begin)(
                    job ) != nullptr ) &&
                (  ::NS(TrackJob_get_particle_set_indices_end)(
                    job ) != nullptr ) &&
                ( *( ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) ==
                        size_t{ 0 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_end)( job ) !=
                ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) &&
                (  ::NS(TrackJob_get_particle_set_index)( job, size_t{ 0 } ) ==
                        size_t{ 0 } ) &&
                (  ::NS(TrackJob_has_elem_by_elem_output)( job ) ) &&
                (  ::NS(TrackJob_has_elem_by_elem_config)( job ) ) &&
                (  ::NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
                    job ) == size_t{ 0 } ) );
        }

        if( success )
        {
            elem_by_elem_conf = ::NS(TrackJob_get_elem_by_elem_config)( job );

            success = ( ( elem_by_elem_conf != nullptr ) &&
                ( ::NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
                ( static_cast< size_t >(
                    ::NS(ElemByElemConfig_get_out_store_num_particles)(
                        elem_by_elem_conf ) ) >=
                    ( NUM_PARTICLES * NUM_BEAM_ELEMENTS
                        * target_num_elem_by_elem_turns ) ) &&
                ( ::NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
                  ::NS(TrackJob_get_default_elem_by_elem_config_rolling_flag)(
                        job ) ) );
        }

        if( success )
        {
            output_buffer = ::NS(TrackJob_get_const_output_buffer)( job );

            success = ( ( output_buffer != nullptr ) &&
                ( ::NS(Buffer_get_num_of_objects)(
                    output_buffer ) == size_t{ 1 } ) &&
                ( ::NS(Buffer_is_particles_buffer)( output_buffer ) ) &&
                ( ::NS(Particles_get_num_of_particles)(
                    ::NS(Particles_buffer_get_const_particles)(
                        output_buffer, size_t{ 0 } ) ) >=
                  ::NS(ElemByElemConfig_get_out_store_num_particles)(
                    elem_by_elem_conf ) ) &&
                (  ::NS(TrackJob_get_beam_monitor_output_buffer_offset)(
                    job ) == size_t{ 1 } ) &&
                (  ::NS(TrackJob_get_beam_monitor_output_buffer_offset)(
                    job ) >= ::NS(Buffer_get_num_of_objects)( output_buffer )
                ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = (
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) &&
                    (  ::NS(TrackJob_get_const_output_buffer)(
                        job ) == ext_output_buffer ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
            else
            {
                success = (
                    ( ::NS(TrackJob_owns_output_buffer)( job ) ) &&
                    ( ::NS(TrackJob_get_const_output_buffer)(
                        job ) != nullptr ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    (  ::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
        }

        if( success )
        {
            success = ( ( ::NS(TrackJob_get_const_particles_buffer)(
                            job ) == particles_buffer ) &&
                        ( ::NS(TrackJob_get_const_beam_elements_buffer)(
                            job ) == beam_elements_buffer ) );
        }

        return success;
    }

    bool test3_CreateTrackJobFullOutput(
        const ::NS(TrackJobCpu) *const SIXTRL_RESTRICT job,
        const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
        ::NS(buffer_size_t) const num_beam_monitors,
        ::NS(buffer_size_t) const target_num_output_turns,
        ::NS(buffer_size_t) const target_num_elem_by_elem_turns )
    {
        bool success = false;

        using size_t              = ::st_buffer_size_t;
        using buffer_t            = ::st_Buffer;
        using particles_t         = ::st_Particles;
        using elem_by_elem_conf_t = ::NS(ElemByElemConfig);

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        SIXTRL_ASSERT( particles_buffer != nullptr );
        SIXTRL_ASSERT( beam_elements_buffer != nullptr );

        particles_t const* particles =
        ::NS(Particles_buffer_get_const_particles)(
                particles_buffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS =
            ::NS(Buffer_get_num_of_objects)( beam_elements_buffer );

        size_t const NUM_PARTICLES =
            ::NS(Particles_get_num_of_particles)( particles );

        success = ( ( job != nullptr ) &&
            ( NUM_BEAM_ELEMENTS > size_t{ 0 } ) &&
            ( NUM_PARTICLES > size_t{ 0 } ) &&
            (  ::NS(TrackJob_has_output_buffer)(  job ) ) &&
            (  ::NS(TrackJob_has_beam_monitor_output)( job ) ) &&
            (  ::NS(TrackJob_get_num_beam_monitors)( job ) ==
                num_beam_monitors ) );

        if( success )
        {
            success = (
                ( ::NS(TrackJob_get_num_particle_sets)(
                    job ) == size_t{ 1 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_begin)(
                    job ) != nullptr ) &&
                (  ::NS(TrackJob_get_particle_set_indices_end)(
                    job ) != nullptr ) &&
                ( *( ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) ==
                        size_t{ 0 } ) &&
                ( ::NS(TrackJob_get_particle_set_indices_end)( job ) !=
                ::NS(TrackJob_get_particle_set_indices_begin)( job ) ) &&
                (  ::NS(TrackJob_get_particle_set_index)( job, size_t{ 0 } ) ==
                        size_t{ 0 } ) );
        }

        if( ( success ) && ( num_beam_monitors > size_t{ 0 } ) )
        {
            size_t const* be_mon_idx_it  =
                ::NS(TrackJob_get_beam_monitor_indices_begin)( job );

            size_t const* be_mon_idx_end =
                ::NS(TrackJob_get_beam_monitor_indices_end)( job );

            success = ( ( job != nullptr ) && ( be_mon_idx_it  != nullptr ) &&
                ( be_mon_idx_end != nullptr ) &&
                ( ::NS(TrackJob_get_beam_monitor_output_buffer_offset)( job )
                    >= ::NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
                    job ) ) &&
                ( std::ptrdiff_t{ 0 } < std::distance(
                    be_mon_idx_it, be_mon_idx_end ) ) &&
                ( static_cast< size_t >( std::distance(
                    be_mon_idx_it, be_mon_idx_end ) ) == num_beam_monitors ) );

            if( success )
            {
                for( ; be_mon_idx_it != be_mon_idx_end ; ++be_mon_idx_it )
                {
                    success &= ( ::NS(OBJECT_TYPE_BEAM_MONITOR) ==
                        NS(Object_get_type_id)( NS(Buffer_get_const_object)(
                            beam_elements_buffer, *be_mon_idx_it ) ) );
                }
            }
        }

        if( ( success ) && ( target_num_elem_by_elem_turns > size_t{ 0 } ) )
        {
            elem_by_elem_conf = ::NS(TrackJob_get_elem_by_elem_config)( job );

            success = (
                (  ::NS(TrackJob_has_elem_by_elem_output)( job ) ) &&
                (  ::NS(TrackJob_has_elem_by_elem_config)( job ) ) &&
                (  ::NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
                    job ) == size_t{ 0 } ) &&
                ( elem_by_elem_conf != nullptr ) &&
                ( ::NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
                ( static_cast< size_t >(
                    ::NS(ElemByElemConfig_get_out_store_num_particles)(
                        elem_by_elem_conf ) ) >=
                    ( NUM_PARTICLES * NUM_BEAM_ELEMENTS
                        * target_num_elem_by_elem_turns ) ) &&
                ( ::NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
                  ::NS(TrackJob_get_default_elem_by_elem_config_rolling_flag)(
                        job ) ) );
        }

        if( ( success ) &&
            ( ( target_num_elem_by_elem_turns > size_t{ 0 } ) ||
              ( target_num_output_turns > size_t{ 0 } ) ||
              ( num_beam_monitors > size_t{ 0 } ) ) )
        {
            output_buffer = ::NS(TrackJob_get_const_output_buffer)( job );

            size_t requ_num_output_elems = num_beam_monitors;

            if( ::NS(TrackJob_has_elem_by_elem_output)( job ) )
            {
                requ_num_output_elems += size_t{ 1 };
            }

            success = ( ( output_buffer != nullptr ) &&
                ( ::NS(Buffer_get_num_of_objects)(
                    output_buffer ) == requ_num_output_elems ) &&
                ( ::NS(Buffer_is_particles_buffer)( output_buffer ) ) );
        }

        if( ( success ) && ( target_num_elem_by_elem_turns > size_t{ 0 } ) &&
            ( elem_by_elem_conf != nullptr ) )
        {
            success = (
                ( ::NS(Particles_get_num_of_particles)(
                    ::NS(Particles_buffer_get_const_particles)( output_buffer,
                    ::NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
                        job ) ) ) >=
                  ::NS(ElemByElemConfig_get_out_store_num_particles)(
                    elem_by_elem_conf ) ) );
        }


        if( ( success ) && ( target_num_output_turns > size_t{ 0 } ) &&
            ( num_beam_monitors > size_t{ 0 } ) )
        {
            success = (
                (  ::NS(TrackJob_get_beam_monitor_output_buffer_offset)(
                        job ) >=
                    ::NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
                        job ) ) &&
                (  ::NS(TrackJob_get_beam_monitor_output_buffer_offset)(
                    job ) < ::NS(Buffer_get_num_of_objects)( output_buffer )
                ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = (
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) &&
                    (  ::NS(TrackJob_get_const_output_buffer)(
                        job ) == ext_output_buffer ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
            else
            {
                success = (
                    ( ::NS(TrackJob_owns_output_buffer)( job ) ) &&
                    ( ::NS(TrackJob_get_const_output_buffer)(
                        job ) != nullptr ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    (  ::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
        }

        if( success )
        {
            success = (
                ( ::NS(TrackJob_get_const_particles_buffer)( job ) ==
                    particles_buffer ) &&
                ( ::NS(TrackJob_get_const_beam_elements_buffer)( job ) ==
                    beam_elements_buffer ) );
        }

        return success;
    }
}
}

/* end: tests/sixtracklib/common/test_track_job_cpu_c99.cpp */
