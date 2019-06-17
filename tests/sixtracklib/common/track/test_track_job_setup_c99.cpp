#include "sixtracklib/common/track/track_job_cpu.h"

#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"

TEST( C99_Cpu_CpuTrackJobSetupTests, CreateTrackJobNoOutput )
{
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = ::NS(CpuTrackJob);
    using size_t        = ::NS(ctrl_size_t);
    using c_buffer_t    = ::NS(Buffer);
    using status_t      = ::NS(ctrl_status_t);
    using particles_t   = ::NS(Particles);

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t const* orig_particles =
        ::NS(Particles_buffer_get_const_particles)(
            in_particle_buffer, size_t{ 0 } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)( pb, orig_particles );
    SIXTRL_ASSERT( particles != nullptr );

    size_t const NUM_ELEM_BY_ELEM_TURNS = size_t{ 0 };

    /* ===================================================================== */

    track_job_t* job0 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job0 ) ==
                 st::ARCHITECTURE_CPU );

    ASSERT_TRUE( 0 == std::strcmp( ::NS(TrackJobNew_get_arch_string)( job0 ),
                                   SIXTRL_ARCHITECTURE_CPU_STR ) );

    status_t status = ::NS(TrackJobNew_reset)( job0, pb, eb, nullptr );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_required_output)(
        job0, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job1 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job1 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job1 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_required_output)(
        job1, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t* job2 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job2 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job2 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_particle_set)(
        job2, pb, good_particle_sets[ 0 ], eb, nullptr );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_required_output)(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job3 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job3 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job3 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job3, pb,
        st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(),
        eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_required_output)(
        job3, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
            st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job4 = ::NS(CpuTrackJob_new_with_output)( pb, eb, my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job4 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job4 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_required_output)(
        job4, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
            st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb,
                my_output_buffer ) );

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(TrackJobNew_delete)( job0 );
    ::NS(TrackJobNew_delete)( job1 );
    ::NS(TrackJobNew_delete)( job2 );
    ::NS(TrackJobNew_delete)( job3 );
    ::NS(TrackJobNew_delete)( job4 );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( my_output_buffer );
    ::NS(Buffer_delete)( in_particle_buffer );
}

TEST( C99_Cpu_CpuTrackJobSetupTests, CreateTrackJobElemByElemOutput )
{
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = ::NS(CpuTrackJob);
    using size_t        = ::NS(ctrl_size_t);
    using c_buffer_t      = ::NS(Buffer);
    using status_t      = ::NS(ctrl_status_t);
    using particles_t   = ::NS(Particles);

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t const* orig_particles =
        ::NS(Particles_buffer_get_const_particles)(
            in_particle_buffer, size_t{ 0 } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)( pb, orig_particles );
    SIXTRL_ASSERT( particles != nullptr );

    size_t const NUM_BEAM_ELEMENTS      = ::NS(Buffer_get_num_of_objects)( eb );
    size_t const NUM_PARTICLES          = ::NS(Particles_get_num_of_particles)( particles );
    size_t const NUM_ELEM_BY_ELEM_TURNS = size_t{  5u };

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    /* ===================================================================== */

    track_job_t* job0 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job0 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( 0 == std::strcmp( ::NS(TrackJobNew_get_arch_string)( job0 ),
                                   SIXTRL_ARCHITECTURE_CPU_STR ) );

    status_t status = ::NS(TrackJobNew_reset_with_output)(
        job0, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
        job0, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job1 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job1 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job1 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
        job1, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t* job2 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job2 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job2 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job2, pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job3 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job3 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job3 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job3, pb,
        st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(),
        eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
        job3, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
            st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
                NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job4 = ::NS(CpuTrackJob_new_with_output)( pb, eb, my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job4 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job4 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
        job4, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
            st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb,
                my_output_buffer, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(TrackJobNew_delete)( job0 );
    ::NS(TrackJobNew_delete)( job1 );
    ::NS(TrackJobNew_delete)( job2 );
    ::NS(TrackJobNew_delete)( job3 );
    ::NS(TrackJobNew_delete)( job4 );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( my_output_buffer );
    ::NS(Buffer_delete)( in_particle_buffer );
}

TEST( C99_Cpu_CpuTrackJobSetupTests, CreateTrackJobBeamMonitor )
{
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = ::NS(CpuTrackJob);
    using size_t        = ::NS(ctrl_size_t);
    using c_buffer_t      = ::NS(Buffer);
    using status_t      = ::NS(ctrl_status_t);
    using particles_t   = ::NS(Particles);
    using be_monitor_t  = ::NS(BeamMonitor);

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t const* orig_particles =
        ::NS(Particles_buffer_get_const_particles)(
            in_particle_buffer, size_t{ 0 } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)( pb, orig_particles );
    SIXTRL_ASSERT( particles != nullptr );

    size_t const NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );
    size_t const NUM_PARTICLES = ::NS(Particles_get_num_of_particles)( particles );

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{   10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{   10 };
    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{    0 };
    size_t const NUM_BEAM_MONITORS       = size_t{    2 };

    be_monitor_t* turn_by_turn_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)( turn_by_turn_monitor, false );
    ::NS(BeamMonitor_set_start)( turn_by_turn_monitor, size_t{ 0 } );
    ::NS(BeamMonitor_set_num_stores)(
        turn_by_turn_monitor, NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( eot_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)(eot_monitor, true );
    ::NS(BeamMonitor_set_start)( eot_monitor, NUM_TURN_BY_TURN_TURNS );
    ::NS(BeamMonitor_set_skip)( eot_monitor, SKIP_TURNS );
    ::NS(BeamMonitor_set_num_stores)( eot_monitor,
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        ::NS(Buffer_get_num_of_objects)( eb ) );

    /* ===================================================================== */

    track_job_t* job0 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job0 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( 0 == std::strcmp( ::NS(TrackJobNew_get_arch_string)( job0 ),
                                   SIXTRL_ARCHITECTURE_CPU_STR ) );

    status_t status = ::NS(TrackJobNew_reset_with_output)(
        job0, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job0, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job1 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job1 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job1 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job1, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t* job2 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job2 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job2 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job2, pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job3 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job3 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job3 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job3, pb,
        st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(),
        eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job3, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job4 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job4 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job4 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job4, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, my_output_buffer,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(TrackJobNew_delete)( job0 );
    ::NS(TrackJobNew_delete)( job1 );
    ::NS(TrackJobNew_delete)( job2 );
    ::NS(TrackJobNew_delete)( job3 );
    ::NS(TrackJobNew_delete)( job4 );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( my_output_buffer );
    ::NS(Buffer_delete)( in_particle_buffer );
}

TEST( C99_Cpu_CpuTrackJobSetupTests, CreateTrackJobBeamMonitorAndElemByElem )
{
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = ::NS(CpuTrackJob);
    using size_t        = ::NS(ctrl_size_t);
    using c_buffer_t    = ::NS(Buffer);
    using status_t      = ::NS(ctrl_status_t);
    using particles_t   = ::NS(Particles);
    using be_monitor_t  = ::NS(BeamMonitor);

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* in_particle_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    particles_t const* orig_particles =
        ::NS(Particles_buffer_get_const_particles)(
            in_particle_buffer, size_t{ 0 } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)( pb, orig_particles );
    SIXTRL_ASSERT( particles != nullptr );

    size_t const NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );
    size_t const NUM_PARTICLES     = ::NS(Particles_get_num_of_particles)( particles );

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{   10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{   10 };
    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{    5 };
    size_t const NUM_BEAM_MONITORS       = size_t{    2 };

    be_monitor_t* turn_by_turn_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)(turn_by_turn_monitor, false );
    ::NS(BeamMonitor_set_start)( turn_by_turn_monitor, size_t{ 0 } );
    ::NS(BeamMonitor_set_num_stores)(
        turn_by_turn_monitor, NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = ::NS(BeamMonitor_new)( eb );
    SIXTRL_ASSERT( eot_monitor != nullptr );

    ::NS(BeamMonitor_set_is_rolling)( eot_monitor, true );
    ::NS(BeamMonitor_set_start)( eot_monitor, NUM_TURN_BY_TURN_TURNS );
    ::NS(BeamMonitor_set_skip)( eot_monitor, SKIP_TURNS );
    ::NS(BeamMonitor_set_num_stores)( eot_monitor,
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS == ::NS(Buffer_get_num_of_objects)( eb ) );

    /* ===================================================================== */

    track_job_t* job0 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job0 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( 0 == std::strcmp( ::NS(TrackJobNew_get_arch_string)( job0 ),
                                   SIXTRL_ARCHITECTURE_CPU_STR ) );

    status_t status = ::NS(TrackJobNew_reset_with_output)(
        job0, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job0, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job1 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job1 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job1 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job1, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t* job2 = ::NS(CpuTrackJob_create)();

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job2 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job2 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job2, pb, size_t{ 1 },
        &good_particle_sets[ 0 ], eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job3 = ::NS(CpuTrackJob_create)();
    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job3 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job3 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    status = ::NS(TrackJobNew_reset_detailed)( job3, pb,
        st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(),
        eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job3, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t* job4 = ::NS(CpuTrackJob_new_with_output)(
        pb, eb, my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( ::NS(TrackJobNew_get_arch_id)( job4 ) == st::ARCHITECTURE_CPU );
    ASSERT_TRUE( std::strcmp( ::NS(TrackJobNew_get_arch_string)( job4 ),
                              SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    ASSERT_TRUE( ::NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
        job4, pb, st::CpuTrackJob::DefaultNumParticleSetIndices(),
        st::CpuTrackJob::DefaultParticleSetIndicesBegin(), eb, my_output_buffer,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(TrackJobNew_delete)( job0 );
    ::NS(TrackJobNew_delete)( job1 );
    ::NS(TrackJobNew_delete)( job2 );
    ::NS(TrackJobNew_delete)( job3 );
    ::NS(TrackJobNew_delete)( job4 );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( my_output_buffer );
    ::NS(Buffer_delete)( in_particle_buffer );
}

/* end: tests/sixtracklib/common/track/test_track_job_setup_c99.cpp */
