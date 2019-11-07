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

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/track_job_cpu.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        bool test1_CreateTrackJobNoOutputDelete(
            st::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
            st::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
            st::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
            st::Buffer const* SIXTRL_RESTRICT ext_output_buffer );

        bool test2_CreateTrackJobElemByElemOutputDelete(
            st::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
            st::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
            st::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
            st::Buffer const* SIXTRL_RESTRICT ext_output_buffer,
            st::Buffer::size_type const target_num_elem_by_elem_turns );

        bool test3_CreateTrackJobFullOutput(
            st::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
            st::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
            st::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
            st::Buffer const* SIXTRL_RESTRICT ext_output_buffer,
            st::Buffer::size_type const num_beam_monitors,
            st::Buffer::size_type const target_num_output_turns,
            st::Buffer::size_type const target_num_elem_by_elem_turns );
    }
}

TEST( CXX_TrackJobCpuTests, CreateTrackJobNoOutputDelete )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::TrackJobCpu;
    using size_t        = st::TrackJobCpu::size_type;
    using buffer_t      = st::Buffer;
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

    /* ===================================================================== *
     * First set of tests:
     * No Beam Monitors
     * No Elem by Elem config
     * --------------------------------------------------------------------- */

    track_job_t job0;

    ASSERT_TRUE( job0.type() == st::TRACK_JOB_CPU_ID );
    ASSERT_TRUE( 0 == job0.typeStr().compare( st::TRACK_JOB_CPU_STR ) );

    bool success = job0.reset( pb, eb, nullptr, size_t{ 0 } );
    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job0, pb, eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( pb, eb, nullptr, size_t{ 0 } );
    ASSERT_TRUE( job1.type() == st::TRACK_JOB_CPU_ID );
    ASSERT_TRUE( job1.typeStr().compare( st::TRACK_JOB_CPU_STR ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job1, pb, eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2( pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb,
                      nullptr, size_t{ 0 } );

    ASSERT_TRUE( job2.type() == st::TRACK_JOB_CPU_ID );
    ASSERT_TRUE( job2.typeStr().compare( st::TRACK_JOB_CPU_STR ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job2, pb, eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    size_t const wrong_sets[] = { size_t{ 0 }, size_t{ 1 }, size_t{ 2 } };

    track_job_t job3( pb, size_t{ 3 }, &wrong_sets[ 0 ],
                      eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( job3.type() == st::TRACK_JOB_CPU_ID );
    ASSERT_TRUE( job3.typeStr().compare( st::TRACK_JOB_CPU_STR ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job3, pb, eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb,
            &my_output_buffer, size_t{ 0 } );

    ASSERT_TRUE( job4.type() == st::TRACK_JOB_CPU_ID );
    ASSERT_TRUE( job4.typeStr().compare( st::TRACK_JOB_CPU_STR ) == 0 );

    ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
        job4, pb, eb, &my_output_buffer ) );
}

TEST( CXX_TrackJobCpuTests, CreateTrackJobElemByElemOutputDelete )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::TrackJobCpu;
    using size_t        = st::TrackJobCpu::size_type;
    using buffer_t      = st::Buffer;
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

    /* ===================================================================== *
     * Second set of tests:
     * No Beam Monitors
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t job0;
    bool success = job0.reset( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job0, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job1( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job1, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2( pb, size_t{ 1 }, &good_particle_sets[ 0 ],
                      eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job2, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const wrong_sets[] = { size_t{ 0 }, size_t{ 1 }, size_t{ 2 } };

    track_job_t job3( pb, size_t{ 3 }, &wrong_sets[ 0 ],
                      eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job3, pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( pb, size_t{ 1 }, &good_particle_sets[ 0 ],
                      eb, &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
        job4, pb, eb, &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS ) );
}

TEST( CXX_TrackJobCpuTests, CreateTrackJobBeamMonitorOutputDelete )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::TrackJobCpu;
    using size_t        = st::TrackJobCpu::size_type;
    using buffer_t      = st::Buffer;
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
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

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

    /* ===================================================================== *
     * Third set of tests:
     * Two Beam Monitors at end of lattice
     * No Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t job0;

    bool success = job0.reset( pb, eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job0, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    /* --------------------------------------------------------------------- */

    track_job_t job1( pb, eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job1, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };
    track_job_t job2( pb, size_t{ 1 }, &good_particle_sets[ 0 ],
                      eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job2, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    /* --------------------------------------------------------------------- */

    size_t const wrong_sets[] = { size_t{ 0 }, size_t{ 1 }, size_t{ 2 } };
    track_job_t job3( pb, size_t{ 3 }, &wrong_sets[ 0 ],
                      eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job3, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb,
                      &my_output_buffer, size_t{ 0 } );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job4, pb, eb,
        &my_output_buffer, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );
}

TEST( CXX_TrackJobCpuTests, CreateTrackJobFullDelete )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::TrackJobCpu;
    using size_t        = st::TrackJobCpu::size_type;
    using buffer_t      = st::Buffer;
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

    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{ 5 };
    size_t const NUM_TURNS               = size_t{ 1000 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    be_monitor_t* turn_by_turn_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( NUM_ELEM_BY_ELEM_TURNS );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( eot_monitor != nullptr );

    eot_monitor->setIsRolling( true );
    eot_monitor->setStart( NUM_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( SKIP_TURNS );
    eot_monitor->setNumStores(
        ( NUM_TURNS - ( NUM_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS ) )
            / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS == eb.getNumObjects() );

    /* ===================================================================== *
     * Fourth set of tests:
     * Two Beam Monitors at end of lattice
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    track_job_t job0;
    bool success = job0.reset( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job0, pb, eb,
        nullptr, NUM_BEAM_MONITORS, NUM_TURNS, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job1( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job1, pb, eb, nullptr,
        NUM_BEAM_MONITORS, NUM_TURNS, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };
    track_job_t job2( pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr,
                      NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job2, pb, eb,
        nullptr, NUM_BEAM_MONITORS, NUM_TURNS, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    size_t const wrong_sets[] = { size_t{ 0 }, size_t{ 1 }, size_t{ 2 } };

    track_job_t job3( pb, size_t{ 3 }, &wrong_sets[ 0 ], eb,
                      nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job3, pb, eb,
        nullptr, NUM_BEAM_MONITORS, NUM_TURNS, NUM_ELEM_BY_ELEM_TURNS ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb,
        &my_output_buffer, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
        job4, pb, eb, &my_output_buffer, NUM_BEAM_MONITORS,
        NUM_TURNS, NUM_ELEM_BY_ELEM_TURNS ) );
}

TEST( CXX_TrackJobCpuTests, TrackParticles )
{
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::TrackJobCpu;
    using size_t        = st::TrackJobCpu::size_type;
    using buffer_t      = st::Buffer;
    using particles_t   = st::Particles;
    using be_monitor_t  = st::BeamMonitor;
    using part_index_t  = ::NS(particle_index_t);

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

    buffer_t cmp_pb;
    buffer_t cmp_output_buffer;

    particles_t* cmp_particles = cmp_pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( cmp_particles != nullptr );
    cmp_particles->copy( *orig_particles );

    size_t const NUM_BEAM_ELEMENTS = eb.getNumObjects();
    size_t const NUM_PARTICLES     = particles->getNumParticles();

    ASSERT_TRUE( NUM_PARTICLES     > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    size_t const NUM_ELEM_BY_ELEM_TURNS  = size_t{ 5 };
    size_t const NUM_TURNS               = size_t{ 100 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    /* --------------------------------------------------------------------- */

    be_monitor_t* turn_by_turn_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( NUM_ELEM_BY_ELEM_TURNS );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );

    be_monitor_t* eot_monitor = eb.createNew< be_monitor_t >();
    SIXTRL_ASSERT( eot_monitor != nullptr );

    eot_monitor->setIsRolling( true );
    eot_monitor->setStart( NUM_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( SKIP_TURNS );
    eot_monitor->setNumStores(
        ( NUM_TURNS - ( NUM_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS ) )
            / SKIP_TURNS );

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS == eb.getNumObjects() );

    /* -------------------------------------------------------------------- */
    /* create cmp particle and output data to verify track job performance  */

    size_t elem_by_elem_offset = size_t{ 0 };
    size_t beam_monitor_offset = size_t{ 0 };
    size_t num_elem_by_elem_objs = size_t{ 0 };

    part_index_t const start_elem = part_index_t{ 0u };
    part_index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
                 min_turn_id, max_turn_id;

    int ret = ::NS(OutputBuffer_get_min_max_attributes)(
        cmp_particles->getCApiPtr(), eb.getCApiPtr(),
        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
        &min_turn_id, &max_turn_id, &num_elem_by_elem_objs, start_elem );

    part_index_t const max_elem_by_elem_turn_id = (
        ( NUM_ELEM_BY_ELEM_TURNS > static_cast< size_t >( min_turn_id ) ) &&
        ( min_turn_id >= part_index_t{ 0 } ) )
        ? ( NUM_ELEM_BY_ELEM_TURNS - 1 ) : min_turn_id;

    ::NS(ElemByElemConfig) config;
    ::NS(ElemByElemConfig_preset)( &config );

    if( ( NS(TRACK_SUCCESS) == ret ) &&
        ( static_cast< size_t >( min_turn_id ) < NUM_ELEM_BY_ELEM_TURNS ) &&
        ( num_elem_by_elem_objs > size_t{ 0 } ) )
    {
        ret = ::NS(ElemByElemConfig_init_detailed)( &config,
            NS(ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES), min_part_id,
                max_part_id, min_elem_id, max_elem_id,
                    min_turn_id, max_elem_by_elem_turn_id, true );
    }
    else if( ret == NS(TRACK_SUCCESS) )
    {
        ret = ::NS(TRACK_STATUS_GENERAL_FAILURE);
    }

    ret = ::NS(OutputBuffer_prepare)( eb.getCApiPtr(),
        cmp_output_buffer.getCApiPtr(), cmp_particles->getCApiPtr(),
            NUM_ELEM_BY_ELEM_TURNS, &elem_by_elem_offset,
                &beam_monitor_offset, &min_turn_id );

    if( ret == ::NS(TRACK_SUCCESS) )
    {
        ret = ::NS(BeamMonitor_assign_output_buffer_from_offset)(
            eb.getCApiPtr(), cmp_output_buffer.getCApiPtr(),
                min_turn_id, beam_monitor_offset );
    }

    if( ret == ::NS(TRACK_SUCCESS) )
    {
        ::NS(ElemByElemConfig_set_output_store_address)(
            &config, ( uintptr_t )::NS(Particles_buffer_get_const_particles)(
                cmp_output_buffer.getCApiPtr(), elem_by_elem_offset ) );
    }

    ret = ::NS(Track_all_particles_element_by_element_until_turn)(
        cmp_particles->getCApiPtr(), &config, eb.getCApiPtr(),
        NUM_ELEM_BY_ELEM_TURNS );

    SIXTRL_ASSERT( ret == NS(TRACK_SUCCESS) );

    ret = ::NS(Track_all_particles_until_turn)(
        cmp_particles->getCApiPtr(), eb.getCApiPtr(), NUM_TURNS );

    SIXTRL_ASSERT( ret == NS(TRACK_SUCCESS) );

    st::BeamMonitor::clearAll( eb );

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    track_job_t job( pb, eb, nullptr, NUM_ELEM_BY_ELEM_TURNS );

    ASSERT_TRUE( job.hasOutputBuffer() );
    ASSERT_TRUE( job.ownsOutputBuffer() );
    ASSERT_TRUE( job.ptrOutputBuffer() != nullptr );
    ASSERT_TRUE( job.hasBeamMonitorOutput() );
    ASSERT_TRUE( job.numBeamMonitors() == NUM_BEAM_MONITORS );

    ASSERT_TRUE( job.hasElemByElemOutput() );
    ASSERT_TRUE( job.ptrElemByElemConfig() != nullptr );

    ret = st::trackElemByElem( job, NUM_ELEM_BY_ELEM_TURNS );
    ASSERT_TRUE( ret == 0 );

    ret = st::trackUntil( job, NUM_TURNS );
    ASSERT_TRUE( ret == 0 );

    st::collect( job );

    /* --------------------------------------------------------------------- */
    /* compare */

    buffer_t const* ptr_output_buffer = job.ptrOutputBuffer();

    ASSERT_TRUE( ptr_output_buffer != nullptr );
    ASSERT_TRUE( ptr_output_buffer->getNumObjects() ==
                 cmp_output_buffer.getNumObjects() );

    double const ABS_ERR = double{ 1e-14 };

    if( ::NS(Particles_buffers_compare_values)( ptr_output_buffer->getCApiPtr(),
            cmp_output_buffer.getCApiPtr() ) != 0 )
    {
        if( ::NS(Particles_buffers_compare_values_with_treshold)(
            ptr_output_buffer->getCApiPtr(), cmp_output_buffer.getCApiPtr(),
                ABS_ERR ) != 0 )
        {
            size_t const nn = cmp_output_buffer.getNumObjects();

            buffer_t diff_buffer;

            for( size_t ii =  size_t{ 0 } ; ii < nn ; ++ii )
            {
                particles_t const* cmp = particles_t::FromBuffer(
                    cmp_output_buffer, ii );

                particles_t const* trk_particles = particles_t::FromBuffer(
                    *ptr_output_buffer, ii );

                ASSERT_TRUE( trk_particles != nullptr );
                ASSERT_TRUE( cmp != nullptr );
                ASSERT_TRUE( cmp->getNumParticles() > 0 );

                ASSERT_TRUE( trk_particles->getNumParticles() ==
                             cmp->getNumParticles() );

                if( 0 != ::NS(Particles_compare_values_with_treshold)(
                        cmp->getCApiPtr(), trk_particles->getCApiPtr(),
                            ABS_ERR ) )
                {
                    particles_t* diff = diff_buffer.createNew< particles_t >(
                        cmp->getNumParticles() );

                    SIXTRL_ASSERT( diff != nullptr );

                    ::NS(Particles_calculate_difference)(
                        cmp->getCApiPtr(), trk_particles->getCApiPtr(),
                        diff->getCApiPtr() );

                    std::cout << "ii = " << ii << std::endl;
                    ::NS(Particles_print_out)( diff->getCApiPtr() );
                    std::cout << std::endl;
                }
            }
        }
    }

    ASSERT_TRUE( ( ::NS(Particles_buffers_compare_values)(
                        ptr_output_buffer->getCApiPtr(),
                        cmp_output_buffer.getCApiPtr() ) == 0 ) ||
                 ( ::NS(Particles_buffers_compare_values_with_treshold)(
                        ptr_output_buffer->getCApiPtr(),
                        cmp_output_buffer.getCApiPtr(), ABS_ERR ) == 0 ) );
}

namespace SIXTRL_CXX_NAMESPACE
{
namespace tests
{
    bool test1_CreateTrackJobNoOutputDelete(
        SIXTRL_CXX_NAMESPACE::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const* SIXTRL_RESTRICT ext_output_buffer )
    {
        using buffer_t = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_t   = buffer_t::size_type;

        bool success = (
            ( ( ( ext_output_buffer == nullptr ) &&
                ( !job.hasOutputBuffer() ) && ( !job.ownsOutputBuffer() ) ) ||
              ( ( ext_output_buffer != nullptr ) &&
                (  job.hasOutputBuffer() ) &&
                ( !job.ownsOutputBuffer() ) ) ) &&
            ( !job.hasBeamMonitorOutput() ) &&
            (  job.numBeamMonitors() == size_t{ 0 } ) );

        if( success )
        {
            success = ( ( job.beamMonitorIndicesBegin() == nullptr ) &&
                        ( job.beamMonitorIndicesEnd()   == nullptr ) );
        }

        if( success )
        {
            success = ( (  job.numParticleSets() == size_t{ 1 } ) &&
                (  job.particleSetIndicesBegin() != nullptr ) &&
                (  job.particleSetIndicesEnd()   != nullptr ) &&
                ( *job.particleSetIndicesBegin() == size_t{ 0 } ) &&
                (  job.particleSetIndicesEnd()   !=
                   job.particleSetIndicesBegin() ) &&
                (  job.particleSetIndex( size_t{ 0 } ) == size_t{ 0 } ) );
        }

        if( success )
        {
            success = ( ( !job.hasElemByElemOutput() ) &&
                ( job.ptrElemByElemConfig() == nullptr ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = ( ( job.ptrOutputBuffer() == ext_output_buffer ) &&
                    ( job.hasOutputBuffer() ) &&
                    ( !job.ownsOutputBuffer() ) );
            }
            else
            {
                success = ( ( job.ptrOutputBuffer() == nullptr ) &&
                    ( !job.hasOutputBuffer() ) &&
                    ( !job.ownsOutputBuffer() ) );
            }
        }

        if( success )
        {
            success = (
                ( job.ptrParticlesBuffer() == &particles_buffer ) &&
                ( job.ptrBeamElementsBuffer() == &beam_elements_buffer ) );
        }

        return success;
    }

    bool test2_CreateTrackJobElemByElemOutputDelete(
        SIXTRL_CXX_NAMESPACE::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const* SIXTRL_RESTRICT ext_output_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type
            const target_num_elem_by_elem_turns )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        bool success = false;

        using buffer_t            = st::Buffer;
        using size_t              = buffer_t::size_type;
        using particles_t         = st::Particles;
        using track_job_t         = st::TrackJobCpu;
        using elem_by_elem_conf_t = track_job_t::elem_by_elem_config_t;

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        SIXTRL_ASSERT( target_num_elem_by_elem_turns > size_t{ 0 } );

        particles_t const* particles = particles_t::FromBuffer(
            particles_buffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS = beam_elements_buffer.getNumObjects();
        size_t const NUM_PARTICLES = particles->getNumParticles();

        success = (
            ( NUM_BEAM_ELEMENTS > size_t{ 0 } ) &&
            ( NUM_PARTICLES > size_t{ 0 } ) &&
            (  job.hasOutputBuffer() ) &&
            ( !job.hasBeamMonitorOutput() ) &&
            (  job.numBeamMonitors() == size_t{ 0 } ) &&
            (  job.beamMonitorIndicesBegin() == nullptr ) &&
            (  job.beamMonitorIndicesEnd()   == nullptr ) );

        if( success )
        {
            success = ( ( job.numParticleSets() == size_t{ 1 } ) &&
                (    job.particleSetIndicesBegin() != nullptr ) &&
                (    job.particleSetIndicesEnd()   != nullptr ) &&
                ( *( job.particleSetIndicesBegin() ) == size_t{ 0 } ) &&
                (    job.particleSetIndicesBegin() !=
                     job.particleSetIndicesEnd() ) &&
                (    job.particleSetIndex( size_t{ 0 } ) == size_t{ 0 } ) &&
                (    job.hasElemByElemOutput() ) &&
                (    job.ptrElemByElemConfig() != nullptr ) &&
                (    job.elemByElemOutputBufferOffset() == size_t{ 0 } ) );
        }

        if( success )
        {
            elem_by_elem_conf = job.ptrElemByElemConfig();

            success = ( ( elem_by_elem_conf != nullptr ) &&
                ( ::NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
                ( static_cast< size_t >(
                    ::NS(ElemByElemConfig_get_out_store_num_particles)(
                        elem_by_elem_conf ) ) >=
                    ( NUM_PARTICLES * NUM_BEAM_ELEMENTS
                        * target_num_elem_by_elem_turns ) ) &&
                ( ::NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
                  job.defaultElemByElemRolling() ) );
        }

        if( success )
        {
            output_buffer = job.ptrOutputBuffer();

            success = ( ( output_buffer != nullptr ) &&
                ( output_buffer->getNumObjects() == size_t{ 1 } ) &&
                ( st::Buffer_is_particles_buffer( *output_buffer ) ) &&
                ( st::Particles::FromBuffer(
                    *output_buffer, size_t{ 0 } )->getNumParticles()  >=
                  ::NS(ElemByElemConfig_get_out_store_num_particles)(
                    elem_by_elem_conf ) ) &&
                (  job.beamMonitorsOutputBufferOffset() == size_t{ 1 } ) &&
                (  job.beamMonitorsOutputBufferOffset() >=
                   output_buffer->getNumObjects() ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = ( ( !job.ownsOutputBuffer() ) &&
                    ( job.ptrOutputBuffer() == ext_output_buffer ) &&
                    ( job.hasOutputBuffer() ) );
            }
            else
            {
                success = ( ( job.ownsOutputBuffer() ) &&
                    ( job.ptrOutputBuffer() != nullptr ) &&
                    ( job.hasOutputBuffer() ) );
            }
        }

        if( success )
        {
            success = (
                ( job.ptrParticlesBuffer() == &particles_buffer ) &&
                ( job.ptrBeamElementsBuffer() == &beam_elements_buffer ) );
        }

        return success;
    }

    bool test3_CreateTrackJobFullOutput(
        SIXTRL_CXX_NAMESPACE::TrackJobCpu const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer const* SIXTRL_RESTRICT ext_output_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const num_beam_monitors,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const target_num_output_turns,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const
            target_num_elem_by_elem_turns )
    {
        bool success = false;

        using buffer_t    = st::Buffer;
        using size_t      = buffer_t::size_type;
        using particles_t = st::Particles;
        using track_job_t = st::TrackJobCpu;
        using elem_by_elem_conf_t = track_job_t::elem_by_elem_config_t;

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        particles_t const* particles = particles_t::FromBuffer(
            particles_buffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS = beam_elements_buffer.getNumObjects();
        size_t const NUM_PARTICLES = particles->getNumParticles();

        success = (
            ( NUM_BEAM_ELEMENTS > size_t{ 0 } ) &&
            ( NUM_PARTICLES > size_t{ 0 } ) &&
            ( job.numBeamMonitors() == num_beam_monitors ) );

        if( success )
        {
            success = (
                (    job.numParticleSets() == size_t{ 1 } ) &&
                (    job.particleSetIndicesBegin() != nullptr ) &&
                (    job.particleSetIndicesEnd()   != nullptr ) &&
                ( *( job.particleSetIndicesBegin() ) == size_t{ 0 } ) &&
                (    job.particleSetIndicesBegin() !=
                     job.particleSetIndicesEnd() ) &&
                (    job.particleSetIndex( size_t{ 0 } ) == size_t{ 0 } ) );
        }

        if( ( success ) && ( num_beam_monitors > size_t{ 0 } ) )
        {
            size_t const* be_mon_idx_it  = job.beamMonitorIndicesBegin();
            size_t const* be_mon_idx_end = job.beamMonitorIndicesEnd();

            success = (
                ( be_mon_idx_it  != nullptr ) &&
                ( be_mon_idx_end != nullptr ) &&
                ( job.beamMonitorsOutputBufferOffset() >=
                  job.elemByElemOutputBufferOffset() ) &&
                ( std::ptrdiff_t{ 0 } < std::distance(
                    be_mon_idx_it, be_mon_idx_end ) ) &&
                ( static_cast< size_t >( std::distance(
                    be_mon_idx_it, be_mon_idx_end ) ) == num_beam_monitors ) );

            if( success )
            {
                for( ; be_mon_idx_it != be_mon_idx_end ; ++be_mon_idx_it )
                {
                    if( ::NS(OBJECT_TYPE_BEAM_MONITOR) !=
                        ::NS(Object_get_type_id)(
                            beam_elements_buffer[ *be_mon_idx_it ] ) )
                    {
                        success = false;
                        break;
                    }
                }
            }
        }

        if( ( success ) && ( target_num_elem_by_elem_turns > size_t{ 0 } ) )
        {
            elem_by_elem_conf = job.ptrElemByElemConfig();

            success = (
                ( job.hasElemByElemOutput() ) &&
                ( job.ptrElemByElemConfig() != nullptr ) &&
                ( job.elemByElemOutputBufferOffset() == size_t{ 0 } ) &&
                ( elem_by_elem_conf != nullptr ) &&
                ( ::NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
                ( static_cast< size_t >(
                    ::NS(ElemByElemConfig_get_out_store_num_particles)(
                        elem_by_elem_conf ) ) >=
                    ( NUM_PARTICLES * NUM_BEAM_ELEMENTS
                        * target_num_elem_by_elem_turns ) ) &&
                ( ::NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
                   job.defaultElemByElemRolling() ) );
        }

        if( ( success ) &&
            ( ( target_num_elem_by_elem_turns > size_t{ 0 } ) ||
              ( target_num_output_turns > size_t{ 0 } ) ||
              ( num_beam_monitors > size_t{ 0 } ) ) )
        {
            output_buffer = job.ptrOutputBuffer();

            size_t requ_num_output_elems = num_beam_monitors;

            if( job.hasElemByElemOutput() )
            {
                requ_num_output_elems += size_t{ 1 };
            }

            success = (
                ( output_buffer != nullptr ) &&
                ( output_buffer->getNumObjects() == requ_num_output_elems ) &&
                ( st::Buffer_is_particles_buffer( *output_buffer ) ) );
        }

        if( ( success ) && ( target_num_elem_by_elem_turns > size_t{ 0 } ) &&
            ( elem_by_elem_conf != nullptr ) )
        {
            success = ( output_buffer->get< st::Particles >(
                size_t{ 0 } )->getNumParticles() >=
                ::NS(ElemByElemConfig_get_out_store_num_particles)(
                    elem_by_elem_conf ) );
        }


        if( ( success ) && ( target_num_output_turns > size_t{ 0 } ) &&
            ( num_beam_monitors > size_t{ 0 } ) )
        {
            success = (
                ( job.beamMonitorsOutputBufferOffset() >=
                  job.elemByElemOutputBufferOffset() ) &&
                ( job.beamMonitorsOutputBufferOffset() <
                  output_buffer->getNumObjects() ) );
        }

        if( success )
        {
            if( ext_output_buffer != nullptr )
            {
                success = (
                    ( !job.ownsOutputBuffer() ) &&
                    (  job.ptrOutputBuffer() == ext_output_buffer ) &&
                    (  job.hasOutputBuffer() ) );
            }
            else
            {
                success = (
                    ( job.ownsOutputBuffer() ) &&
                    ( job.ptrOutputBuffer() != nullptr ) &&
                    ( job.hasOutputBuffer() ) );
            }
        }

        if( success )
        {
            success = (
                ( job.ptrParticlesBuffer() == &particles_buffer ) &&
                ( job.ptrBeamElementsBuffer() == &beam_elements_buffer ) );
        }

        return success;
    }
}
}

/* end: tests/sixtracklib/common/test_track_job_cpu_c99.cpp */
