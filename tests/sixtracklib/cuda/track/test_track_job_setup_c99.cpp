#include "sixtracklib/cuda/track_job.hp"

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

TEST( CXX_CudaTrackJobSetupTests, CreateTrackJobNoOutput )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::CudaTrackJob;
    using size_t        = track_job_t::size_type;
    using buffer_t      = track_job_t::buffer_t;
    using status_t      = track_job_t::status_t;
    using particles_t   = st::Particles;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t pb;
    buffer_t my_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );
    particles->copy( *orig_particles );

    /* ===================================================================== */

    track_job_t job0;

    ASSERT_TRUE( job0.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( 0 == job0.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) );

    status_t status = job0.reset( pb, eb );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job0, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( "0.0", pb, eb, nullptr, size_t{ 0 } );
    ASSERT_TRUE( job1.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job1.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job1, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2;

    ASSERT_TRUE( job2.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job2.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job2.reset( pb, good_particle_sets[ 0 ], eb, nullptr );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t job3;

    ASSERT_TRUE( job3.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job3.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job3.reset( pb, track_job_t::DefaultParticleSetIndicesBegin(),
        track_job_t::DefaultParticleSetIndicesEnd(), eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job3, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( "0.0", pb, eb, &my_output_buffer );

    ASSERT_TRUE( job4.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job4.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job4, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb,
                &my_output_buffer ) );
}

TEST( CXX_CudaTrackJobSetupTests, CreateTrackJobElemByElemOutput )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::CudaTrackJob;
    using size_t        = track_job_t::size_type;
    using buffer_t      = track_job_t::buffer_t;
    using status_t      = track_job_t::status_t;
    using particles_t   = st::Particles;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t pb;
    buffer_t my_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );
    particles->copy( *orig_particles );

    size_t const NUM_BEAM_ELEMENTS      = eb.getNumObjects();
    size_t const NUM_PARTICLES          = particles->getNumParticles();
    size_t const NUM_ELEM_BY_ELEM_TURNS = size_t{  5u };

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    /* ===================================================================== */

    track_job_t job0;

    ASSERT_TRUE( job0.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( 0 == job0.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) );

    status_t status = job0.reset( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_beam_monitors_elem_by_elem(
        job0, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( "0.0", pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( job1.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job1.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_beam_monitors_elem_by_elem(
        job1, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2;

    ASSERT_TRUE( job2.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job2.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job2.reset( pb, good_particle_sets[ 0 ], eb,
                             nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_beam_monitors_elem_by_elem(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job3;

    ASSERT_TRUE( job3.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job3.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job3.reset( pb, track_job_t::DefaultParticleSetIndicesBegin(),
        track_job_t::DefaultParticleSetIndicesEnd(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_beam_monitors_elem_by_elem(
        job3, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
                NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( "0.0", pb, eb, &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job4.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job4.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_beam_monitors_elem_by_elem(
        job4, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb,
                &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS ) );
}

TEST( CXX_CudaTrackJobSetupTests, CreateTrackJobBeamMonitor )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::CudaTrackJob;
    using size_t        = track_job_t::size_type;
    using buffer_t      = track_job_t::buffer_t;
    using status_t      = track_job_t::status_t;
    using particles_t   = st::Particles;
    using be_monitor_t  = st::BeamMonitor;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t pb;
    buffer_t my_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );
    particles->copy( *orig_particles );

    size_t const NUM_BEAM_ELEMENTS = eb.getNumObjects();
    size_t const NUM_PARTICLES     = particles->getNumParticles();

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{   10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{   10 };
    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{    0 };
    size_t const NUM_BEAM_MONITORS       = size_t{    2 };

    be_monitor_t* turn_by_turn_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( size_t{ 0 } );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( eot_monitor != nullptr );

    eot_monitor->setIsRolling( true );
    eot_monitor->setStart( NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( SKIP_TURNS );
    eot_monitor->setNumStores(
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS == eb.getNumObjects() );

    /* ===================================================================== */

    track_job_t job0;

    ASSERT_TRUE( job0.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( 0 == job0.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) );

    status_t status = job0.reset( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job0, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( "0.0", pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( job1.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job1.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job1, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2;

    ASSERT_TRUE( job2.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job2.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job2.reset( pb, good_particle_sets[ 0 ], eb,
                             nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job3;

    ASSERT_TRUE( job3.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job3.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job3.reset( pb, track_job_t::DefaultParticleSetIndicesBegin(),
        track_job_t::DefaultParticleSetIndicesEnd(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job3, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( "0.0", pb, eb, &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job4.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job4.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job4, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, &my_output_buffer,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );
}

TEST( CXX_CudaTrackJobSetupTests, CreateTrackJobBeamMonitorAndElemByElem )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::CudaTrackJob;
    using size_t        = track_job_t::size_type;
    using buffer_t      = track_job_t::buffer_t;
    using status_t      = track_job_t::status_t;
    using particles_t   = st::Particles;
    using be_monitor_t  = st::BeamMonitor;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t pb;
    buffer_t my_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );
    particles->copy( *orig_particles );

    size_t const NUM_BEAM_ELEMENTS = eb.getNumObjects();
    size_t const NUM_PARTICLES     = particles->getNumParticles();

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{   10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{   10 };
    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{    5 };
    size_t const NUM_BEAM_MONITORS       = size_t{    2 };

    be_monitor_t* turn_by_turn_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( size_t{ 0 } );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( eot_monitor != nullptr );

    eot_monitor->setIsRolling( true );
    eot_monitor->setStart( NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( SKIP_TURNS );
    eot_monitor->setNumStores(
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS == eb.getNumObjects() );

    /* ===================================================================== */

    track_job_t job0;

    ASSERT_TRUE( job0.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( 0 == job0.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) );

    status_t status = job0.reset( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job0, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( "0.0", pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( job1.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job1.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job1, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2;

    ASSERT_TRUE( job2.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job2.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job2.reset( pb, good_particle_sets[ 0 ], eb,
                             nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job3;

    ASSERT_TRUE( job3.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job3.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    status = job3.reset( pb, track_job_t::DefaultParticleSetIndicesBegin(),
        track_job_t::DefaultParticleSetIndicesEnd(), eb, nullptr,
            NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job3, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( "0.0", pb, eb, &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job4.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job4.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_beam_monitors_and_elem_by_elem(
        job4, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, &my_output_buffer,
            NUM_BEAM_MONITORS, NUM_TURNS,  NUM_ELEM_BY_ELEM_TURNS ) );
}

/* end: tests/sixtracklib/cuda/track/test_track_job_setup_cxx.cpp */
