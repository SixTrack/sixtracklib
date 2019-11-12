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
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/track_job_cl.h"


namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool test1_CreateTrackJobNoOutputDelete(
            SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
            const SIXTRL_CXX_NAMESPACE::Buffer *const
                SIXTRL_RESTRICT ext_output_buffer );

        bool test2_CreateTrackJobElemByElemOutputDelete(
            SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
            const SIXTRL_CXX_NAMESPACE::Buffer *const
                SIXTRL_RESTRICT ext_output_buffer,
            SIXTRL_CXX_NAMESPACE::Buffer::size_type const
                target_num_elem_by_elem_turns );

        bool test3_CreateTrackJobFullOutput(
            SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
            SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
            const SIXTRL_CXX_NAMESPACE::Buffer *const
                SIXTRL_RESTRICT ext_output_buffer,
            SIXTRL_CXX_NAMESPACE::Buffer::size_type const num_beam_monitors,
            SIXTRL_CXX_NAMESPACE::Buffer::size_type const num_output_turns,
            SIXTRL_CXX_NAMESPACE::Buffer::size_type const
                target_num_elem_by_elem_turns );
    }
}

TEST( CXX_TrackJobClTests, CreateTrackJobNoOutputDelete )
{
    namespace st      = SIXTRL_CXX_NAMESPACE;
    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    using track_job_t      = st::TrackJobCl;
    using buffer_t         = st::Buffer;
    using size_t           = buffer_t::size_type;
    using particles_t      = st::Particles;
    using cl_context_t     = st::ClContext;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;
    using status_t         = particles_t::status_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t pb;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );

    status_t status = particles->copy( *orig_particles );
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    ( void )status;

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    /* ===================================================================== *
     * First set of tests:
     * No Beam Monitors
     * No Elem by Elem config
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t  job0( device_id_str );
        ASSERT_TRUE( job0.type() == ::NS(TRACK_JOB_CL_ID) );
        ASSERT_TRUE( job0.typeStr().compare( ::NS(TRACK_JOB_CL_STR) ) == 0 );

        status = job0.reset( pb, eb, nullptr, size_t{ 0 } )
            ? ::NS(ARCH_STATUS_SUCCESS)
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job0, pb, eb, nullptr ) );

        /* ----------------------------------------------------------------- */

        track_job_t job1( device_id_str, pb, eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( job1.type() == ::NS(TRACK_JOB_CL_ID) );
        ASSERT_TRUE( job1.typeStr().compare( ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job1, pb, eb, nullptr ) );

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        track_job_t job2( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( job2.type() == ::NS(TRACK_JOB_CL_ID) );
        ASSERT_TRUE( job2.typeStr().compare( ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job2, pb, eb, nullptr ) );

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        track_job_t job3( device_id_str, pb, &wrong_particle_sets[ 0 ],
            &wrong_particle_sets[ 3 ], eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( job3.type() == ::NS(TRACK_JOB_CL_ID) );
        ASSERT_TRUE( job3.typeStr().compare( ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job3, pb, eb, nullptr ) );

        /* ----------------------------------------------------------------- */

        buffer_t my_output_buffer;

        track_job_t job4( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, &my_output_buffer, size_t{ 0 } );

        ASSERT_TRUE( job4.type() == ::NS(TRACK_JOB_CL_ID) );
        ASSERT_TRUE( job4.typeStr().compare( ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job4, pb, eb, &my_output_buffer ) );

        /* ================================================================= */
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";
    }

    std::cout << std::endl;
}

TEST( CXX_TrackJobClTests, CreateTrackJobElemByElemOutputDelete )
{
    namespace st      = SIXTRL_CXX_NAMESPACE;
    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    using track_job_t      = st::TrackJobCl;
    using buffer_t         = st::Buffer;
    using size_t           = buffer_t::size_type;
    using particles_t      = st::Particles;
    using cl_context_t     = st::ClContext;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;
    using status_t         = particles_t::status_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t pb;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );

    status_t status = particles->copy( *orig_particles );
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    size_t const NUM_BEAM_ELEMENTS       = eb.getNumObjects();
    size_t const NUM_PARTICLES           = particles->getNumParticles();
    size_t const DUMP_ELEM_BY_ELEM_TURNS = size_t{ 5u };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    /* ===================================================================== *
     * Second set of tests:
     * No Beam Monitors
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t job0( device_id_str );
        status = job0.reset( pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS )
            ? ::NS(ARCH_STATUS_SUCCESS)
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job0, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        track_job_t job1( device_id_str, pb, eb, nullptr,
                          DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job1, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };
        track_job_t job2( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job2, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        track_job_t job3( device_id_str, pb, &wrong_particle_sets[ 0 ],
            &wrong_particle_sets[ 3 ], eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job3, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        buffer_t my_output_buffer;

        track_job_t job4( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, &my_output_buffer,
                DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job4, pb, eb, &my_output_buffer, DUMP_ELEM_BY_ELEM_TURNS ) );
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";
    }

    std::cout << std::endl;
}

TEST( CXX_TrackJobClTests, CreateTrackJobBeamMonitorOutputDelete )
{
    namespace st      = SIXTRL_CXX_NAMESPACE;
    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    using track_job_t      = st::TrackJobCl;
    using buffer_t         = st::Buffer;
    using size_t           = buffer_t::size_type;
    using particles_t      = st::Particles;
    using cl_context_t     = st::ClContext;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;
    using beam_monitor_t   = st::BeamMonitor;
    using status_t         = particles_t::status_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t pb;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );

    status_t status = particles->copy( *orig_particles );
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    size_t const NUM_BEAM_ELEMENTS       = eb.getNumObjects();
    size_t const NUM_PARTICLES           = particles->getNumParticles();
    size_t const NUM_TURNS               = size_t{ 100 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    beam_monitor_t* turn_by_turn_monitor = eb.createNew< beam_monitor_t >();
    ASSERT_TRUE( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( size_t{ 0 } );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );
    turn_by_turn_monitor->setSkip( size_t{ 1 } );

    beam_monitor_t* eot_monitor = eb.createNew< beam_monitor_t >();
    ASSERT_TRUE( eot_monitor != nullptr );

    eot_monitor->setIsRolling( false );
    eot_monitor->setStart( NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( size_t{ 1 } );

    eot_monitor->setNumStores(
        ( NUM_TURNS - NUM_TURN_BY_TURN_TURNS ) / SKIP_TURNS );

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( eb.getNumObjects() ) );

    /* ===================================================================== *
     * Third set of tests:
     * Two Beam Monitors at end of lattice
     * No Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t job0( device_id_str );

        status = job0.reset( pb, eb, nullptr )
            ? ::NS(ARCH_STATUS_SUCCESS)
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job0, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        /* ----------------------------------------------------------------- */

        track_job_t job1( device_id_str, pb, eb );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job1, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        track_job_t job2( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job2, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        track_job_t job3( device_id_str, pb, &wrong_particle_sets[ 0 ],
            &wrong_particle_sets[ 3 ], eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job3, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        /* ----------------------------------------------------------------- */

        buffer_t my_output_buffer;

        track_job_t job4( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, &my_output_buffer, size_t{ 0 } );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job4, pb, eb, &my_output_buffer, NUM_BEAM_MONITORS,
                NUM_TURNS, size_t{ 0 } ) );
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";
    }

    std::cout << std::endl;
}

TEST( CXX_TrackJobClTests, CreateTrackJobFullDelete )
{
    namespace st      = SIXTRL_CXX_NAMESPACE;
    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    using track_job_t      = st::TrackJobCl;
    using buffer_t         = st::Buffer;
    using size_t           = buffer_t::size_type;
    using particles_t      = st::Particles;
    using cl_context_t     = st::ClContext;
    using beam_monitor_t   = st::BeamMonitor;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;
    using status_t         = particles_t::status_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t pb;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );

    status_t status = particles->copy( *orig_particles );
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    ( void )status;

    size_t const NUM_BEAM_ELEMENTS       = eb.getNumObjects();
    size_t const NUM_PARTICLES           = particles->getNumParticles();
    size_t const NUM_TURNS               = size_t{ 100 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const DUMP_ELEM_BY_ELEM_TURNS = size_t{ 5 };
    size_t const NUM_TURN_BY_TURN_TURNS  = size_t{ 10 };
    size_t const NUM_BEAM_MONITORS       = size_t{ 2 };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    beam_monitor_t* turn_by_turn_monitor = eb.createNew< beam_monitor_t >();
    ASSERT_TRUE( turn_by_turn_monitor != nullptr );

    turn_by_turn_monitor->setIsRolling( false );
    turn_by_turn_monitor->setStart( DUMP_ELEM_BY_ELEM_TURNS );
    turn_by_turn_monitor->setNumStores( NUM_TURN_BY_TURN_TURNS );
    turn_by_turn_monitor->setSkip( size_t{ 1 } );

    beam_monitor_t* eot_monitor = eb.createNew< beam_monitor_t >();
    ASSERT_TRUE( eot_monitor != nullptr );

    eot_monitor->setIsRolling( false );
    eot_monitor->setStart( DUMP_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS );
    eot_monitor->setSkip( size_t{ 1 } );

    eot_monitor->setNumStores(
        ( NUM_TURNS - ( DUMP_ELEM_BY_ELEM_TURNS + NUM_TURN_BY_TURN_TURNS ) /
            SKIP_TURNS ) );

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( eb.getNumObjects() ) );

    /* ===================================================================== *
     * Fourth set of tests:
     * Two Beam Monitors at end of lattice
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t job0( device_id_str );
        status_t status = job0.reset(
            pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS )
            ? ::NS(ARCH_STATUS_SUCCESS)
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job0, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        track_job_t job1( device_id_str, pb, eb, nullptr,
                          DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job1, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        track_job_t job2( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job2, pb, eb,
            nullptr, NUM_BEAM_MONITORS, NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        track_job_t job3( device_id_str, pb, &wrong_particle_sets[ 0 ],
            &wrong_particle_sets[ 0 ], eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job3, pb, eb,
            nullptr, NUM_BEAM_MONITORS, NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

        /* ----------------------------------------------------------------- */

        buffer_t my_output_buffer;

        track_job_t job4( device_id_str, pb, &good_particle_sets[ 0 ],
            &good_particle_sets[ 1 ], eb, &my_output_buffer,
                DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput( job4, pb, eb,
            &my_output_buffer, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";
    }

    std::cout << std::endl;
}

TEST( CXX_TrackJobClTests, CreateTrackJobTrackLineCompare )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using buffer_t         = st::Buffer;
    using particles_t      = st::Particles;
    using buf_size_t       = buffer_t::size_type;
    using track_job_t      = st::TrackJobCl;
    using cl_context_t     = st::ClContext;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;

    buffer_t pb( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );


    particles_t* orig_particles =
        st::Particles::FromBuffer( pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    buf_size_t const num_particles = orig_particles->getNumParticles();

    buffer_t cmp_track_pb;
    particles_t* cmp_particles  =
        cmp_track_pb.createNew< particles_t >( num_particles );

    SIXTRL_ASSERT( cmp_particles != nullptr );
    cmp_particles->copy( *orig_particles );

    buf_size_t const until_turn = 10;
    int status = ::NS(Track_all_particles_until_turn)(
        cmp_particles->getCApiPtr(), eb.getCApiPtr(), until_turn );

    SIXTRL_ASSERT( status == 0 );

    buf_size_t const num_beam_elements = eb.getNumObjects();
    buf_size_t const num_lattice_parts = buf_size_t{ 10 };
    buf_size_t const num_elem_per_part = num_beam_elements / num_lattice_parts;

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    buf_size_t const num_nodes = context.numAvailableNodes();

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        buffer_t track_pb;
        particles_t* particles =
            track_pb.createNew< particles_t >( num_particles );

        SIXTRL_ASSERT( particles != nullptr );
        particles->copy( *orig_particles );
        ::NS(BeamMonitor_clear_all)( eb.getCApiPtr() );

        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );
        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t job( device_id_str, track_pb, eb );

        for( buf_size_t ii = buf_size_t{ 0 } ; ii < until_turn ; ++ii )
        {
            buf_size_t jj = buf_size_t{ 0 };

            for( ; jj < num_lattice_parts ; ++jj )
            {
                bool const is_last_in_turn = ( jj == ( num_lattice_parts - 1 ) );
                buf_size_t const begin_idx =  jj * num_elem_per_part;
                buf_size_t const end_idx   = ( !is_last_in_turn ) ?
                    begin_idx + num_elem_per_part : num_beam_elements;

                status = st::trackLine(
                    job, begin_idx, end_idx, is_last_in_turn );

                 ASSERT_TRUE( status == 0 );
            }
        }

        job.collect();

        double const ABS_DIFF = double{ 2e-14 };

        if( ( 0 != ::NS(Particles_compare_values)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) &&
            ( 0 != ::NS(Particles_compare_values_with_treshold)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr(), ABS_DIFF ) ) )
        {
            buffer_t diff_buffer;

            particles_t* diff =
                diff_buffer.createNew< particles_t >( num_particles );

            SIXTRL_ASSERT( diff != nullptr );

            ::NS(Particles_calculate_difference)( particles->getCApiPtr(),
                cmp_particles->getCApiPtr(), diff->getCApiPtr() );

            printf( "particles: \r\n" );
            ::NS(Particles_print_out)( particles->getCApiPtr() );

            printf( "cmp_particles: \r\n" );
            ::NS(Particles_print_out)( cmp_particles->getCApiPtr() );

            printf( "diff: \r\n" );
            ::NS(Particles_print_out)( diff->getCApiPtr() );
        }

        ASSERT_TRUE(
            ( 0 == ::NS(Particles_compare_values)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr(), ABS_DIFF ) ) );
    }
}

TEST( CXX_TrackJobClTests, TrackParticles )
{
    namespace st      = SIXTRL_CXX_NAMESPACE;
    namespace st_test = SIXTRL_CXX_NAMESPACE::tests;

    using track_job_t      = st::TrackJobCl;
    using buffer_t         = st::Buffer;
    using size_t           = buffer_t::size_type;
    using particles_t      = st::Particles;
    using cl_context_t     = st::ClContext;
    using node_info_t      = cl_context_t::node_info_t;
    using node_id_t        = cl_context_t::node_id_t;
    using node_info_iter_t = node_info_t const*;
    using part_index_t     = particles_t::index_t;
    using beam_monitor_t   = st::BeamMonitor;
    using status_t         = particles_t::status_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t pb;

    buffer_t cmp_pb;
    buffer_t cmp_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );

    status_t status = particles->copy( *orig_particles );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    particles_t* cmp_particles = cmp_pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    status = cmp_particles->copy( *orig_particles );
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    size_t const NUM_BEAM_ELEMENTS       = eb.getNumObjects();
    size_t const NUM_PARTICLES           = particles->getNumParticles();
    size_t const UNTIL_TURN              = size_t{ 100 };
    size_t const SKIP_TURNS              = size_t{ 10 };
    size_t const UNTIL_TURN_ELEM_BY_ELEM = size_t{ 5 };
    size_t const UNTIL_TURN_TURN_BY_TURN = size_t{ 10 };
    size_t NUM_BEAM_MONITORS             = size_t{ 0 };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    if( ( UNTIL_TURN_TURN_BY_TURN > UNTIL_TURN_ELEM_BY_ELEM ) &&
        ( UNTIL_TURN_TURN_BY_TURN > size_t{ 0 } ) )
    {
        beam_monitor_t* turn_by_turn_monitor = eb.createNew< beam_monitor_t >();
        ASSERT_TRUE( turn_by_turn_monitor != nullptr );

        turn_by_turn_monitor->setIsRolling( false );
        turn_by_turn_monitor->setStart( UNTIL_TURN_ELEM_BY_ELEM );
        turn_by_turn_monitor->setNumStores(
            UNTIL_TURN_TURN_BY_TURN - UNTIL_TURN_ELEM_BY_ELEM );
        turn_by_turn_monitor->setSkip( size_t{ 1 } );

        ++NUM_BEAM_MONITORS;
    }

    if( UNTIL_TURN > UNTIL_TURN_TURN_BY_TURN )
    {
        beam_monitor_t* eot_monitor = eb.createNew< beam_monitor_t >();
        ASSERT_TRUE( eot_monitor != nullptr );

        eot_monitor->setIsRolling( false );
        eot_monitor->setStart( UNTIL_TURN_TURN_BY_TURN );
        eot_monitor->setSkip( SKIP_TURNS );

        eot_monitor->setNumStores(
            ( UNTIL_TURN - UNTIL_TURN_TURN_BY_TURN ) / SKIP_TURNS );

        ++NUM_BEAM_MONITORS;
    }

    /* -------------------------------------------------------------------- */
    /* create cmp particle and output data to verify track job performance  */

    size_t elem_by_elem_offset = size_t{ 0 };
    size_t beam_monitor_offset = size_t{ 0 };
    part_index_t min_turn_id   = part_index_t{ 0 };

    int ret = ::NS(OutputBuffer_prepare)(
        eb.getCApiPtr(), cmp_output_buffer.getCApiPtr(),
        cmp_particles->getCApiPtr(), UNTIL_TURN_ELEM_BY_ELEM,
        &elem_by_elem_offset, &beam_monitor_offset, &min_turn_id );

    SIXTRL_ASSERT( ret == 0 );

    if( NUM_BEAM_MONITORS > size_t{ 0u } )
    {
        ret = ::NS(BeamMonitor_assign_output_buffer_from_offset)(
            eb.getCApiPtr(), cmp_output_buffer.getCApiPtr(),
                min_turn_id, beam_monitor_offset );

        ASSERT_TRUE( ret == 0 );
    }

    SIXTRL_ASSERT( ret == 0 );

    if( UNTIL_TURN_ELEM_BY_ELEM > size_t{ 0 } )
    {
        ::NS(ElemByElemConfig) config;
        ::NS(ElemByElemConfig_preset)( &config );

        ret = ::NS(ElemByElemConfig_init)( &config,
            cmp_particles->getCApiPtr(), eb.getCApiPtr(),
                part_index_t{ 0 }, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );

        ret = ::NS(ElemByElemConfig_assign_output_buffer)(
            &config, cmp_output_buffer.getCApiPtr(), elem_by_elem_offset );

        ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );

        ret = ::NS(Track_all_particles_element_by_element_until_turn)(
            cmp_particles->getCApiPtr(), &config, eb.getCApiPtr(),
                UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( ret == 0 );
    }

    if( UNTIL_TURN > UNTIL_TURN_ELEM_BY_ELEM )
    {
        ret = ::NS(Track_all_particles_until_turn)(
            cmp_particles->getCApiPtr(), eb.getCApiPtr(), UNTIL_TURN );

        SIXTRL_ASSERT( ret == 0 );
    }

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    ASSERT_TRUE( NUM_BEAM_ELEMENTS + NUM_BEAM_MONITORS ==
        static_cast< size_t >( eb.getNumObjects() ) );

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    node_info_iter_t node_it  = context.availableNodesInfoBegin();
    node_info_iter_t node_end = context.availableNodesInfoEnd();
    node_id_t default_node_id = context.defaultNodeId();

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        beam_monitor_t::clearAll( eb );

        ::NS(Particles_copy)( particles->getCApiPtr(), particles_t::FromBuffer(
                in_particle_buffer, size_t{ 0 } )->getCApiPtr() );

        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";
        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        std::string const device_id_str( tmp_device_id_str );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t job( device_id_str, pb, eb, nullptr,
                         UNTIL_TURN_ELEM_BY_ELEM );

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.context().hasSelectedNode() );

        ASSERT_TRUE( job.hasOutputBuffer() );
        ASSERT_TRUE( job.ownsOutputBuffer() );
        ASSERT_TRUE( job.ptrOutputBuffer()  != nullptr );
        ASSERT_TRUE( job.ptrCOutputBuffer() ==
                     job.ptrOutputBuffer()->getCApiPtr() );

        ASSERT_TRUE( job.numBeamMonitors() == NUM_BEAM_MONITORS );

        ASSERT_TRUE( job.hasBeamMonitorOutput() ==
                     ( NUM_BEAM_MONITORS > size_t{ 0 } ) );

        ASSERT_TRUE( job.hasElemByElemOutput() ==
                     ( UNTIL_TURN_ELEM_BY_ELEM > size_t{ 0 } ) );

        ASSERT_TRUE( job.ptrElemByElemConfig() != nullptr );

        if( UNTIL_TURN_ELEM_BY_ELEM > size_t{ 0 } )
        {
            ret = st::trackElemByElem( job, UNTIL_TURN_ELEM_BY_ELEM );
            ASSERT_TRUE( ret == 0 );
        }

        if( UNTIL_TURN > UNTIL_TURN_ELEM_BY_ELEM )
        {
            ret = st::track( job, UNTIL_TURN );
            ASSERT_TRUE( ret == 0 );
        }

        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.particlesArg().usesCObjectBuffer() );
        SIXTRL_ASSERT( job.particlesArg().ptrCObjectBuffer() ==
                       job.ptrCParticlesBuffer() );
        SIXTRL_ASSERT( job.particlesArg().context() == job.ptrContext() );

        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.beamElementsArg().usesCObjectBuffer() );
        SIXTRL_ASSERT( job.beamElementsArg().ptrCObjectBuffer() ==
                       job.ptrCBeamElementsBuffer() );
        SIXTRL_ASSERT( job.beamElementsArg().context() == job.ptrContext() );

        st::collect( job );

        /* --------------------------------------------------------------------- */
        /* compare */

        buffer_t const* ptr_output_buffer = job.ptrOutputBuffer();

        ASSERT_TRUE( ptr_output_buffer != nullptr );
        ASSERT_TRUE( ptr_output_buffer->getNumObjects() ==
                      cmp_output_buffer.getNumObjects() );

        double const ABS_ERR = double{ 2e-14 };

        if( ::NS(Particles_buffers_compare_values)(
                ptr_output_buffer->getCApiPtr(),
                cmp_output_buffer.getCApiPtr() ) != 0 )
        {
            if( ::NS(Particles_buffers_compare_values_with_treshold)(
                ptr_output_buffer->getCApiPtr(),
                cmp_output_buffer.getCApiPtr(), ABS_ERR ) != 0 )
            {
                size_t const nn = cmp_output_buffer.getNumObjects();
                buffer_t diff_buffer;

                for( size_t ii =  size_t{ 0 } ; ii < nn ; ++ii )
                {
                    particles_t const* cmp = particles_t::FromBuffer(
                        cmp_output_buffer, ii );

                    particles_t const* trk_particles = particles_t::FromBuffer(
                        *ptr_output_buffer, ii );

                    size_t const mm = cmp->getNumParticles();

                    ASSERT_TRUE( mm > 0 );
                    ASSERT_TRUE( mm == static_cast< size_t >(
                        trk_particles->getNumParticles() ) );

                    if( 0 != ::NS(Particles_compare_values_with_treshold)(
                            cmp->getCApiPtr(), trk_particles->getCApiPtr(),
                                ABS_ERR ) )
                    {
                        size_t const diff_index = diff_buffer.getNumObjects();

                        particles_t* diff =
                            diff_buffer.createNew< particles_t >( mm );

                        std::vector< ::NS(buffer_size_t) > max_diff_indices(
                            30, ::NS(buffer_size_t){ 0 } );

                        SIXTRL_ASSERT( diff != nullptr );

                        ::NS(Particles_calculate_difference)(
                            cmp->getCApiPtr(), trk_particles->getCApiPtr(),
                            diff->getCApiPtr() );

                        particles_t* max_diff =
                            diff_buffer.createNew< particles_t >( mm );

                        SIXTRL_ASSERT( max_diff != nullptr );

                        ::NS(Particles_get_max_difference)(
                            max_diff->getCApiPtr(), max_diff_indices.data(),
                            cmp->getCApiPtr(), trk_particles->getCApiPtr() );

                        std::cout <<
                                "-------------------------------------------"
                                "-------------------------------------------"
                                "-------------------------------------------"
                                "\r\nparticle set ii = " << ii <<
                                "\r\nmax_diff : \r\n";

                        ::NS(Particles_print_max_diff_out)(
                            max_diff->getCApiPtr(), max_diff_indices.data() );

                        diff = particles_t::FromBuffer(
                            diff_buffer, diff_index );


                        for( size_t jj = size_t{ 0 } ; jj < mm ; ++jj )
                        {
                            std::cout << "\r\nparticle index ii = "
                                      << jj << "\r\ncmp: \r\n";

                            ::NS(Particles_print_out_single)(
                                cmp->getCApiPtr(), jj );

                            std::cout << "\r\ntrk_particles: \r\n";
                            ::NS(Particles_print_out_single)(
                                trk_particles->getCApiPtr(), jj );

                            std::cout << "\r\ndiff: \r\n";
                            ::NS(Particles_print_out_single)(
                                diff->getCApiPtr(), jj );
                            std::cout << std::endl;

                            if( jj < ( mm - size_t{ 1 } ) )
                            {
                                std::cout <<
                                "- - - - - - - - - - - - - - - - - - - - - -"
                                " - - - - - - - - - - - - - - - - - - - - - "
                                "- - - - - - - - - - - - - - - - - - - - - -"
                                "\r\n";
                            }
                        }
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

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";
    }

    std::cout << std::endl;
}
//

namespace SIXTRL_CXX_NAMESPACE
{
namespace tests
{
    bool test1_CreateTrackJobNoOutputDelete(
        SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
        const SIXTRL_CXX_NAMESPACE::Buffer *const
            SIXTRL_RESTRICT ext_output_buffer )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using size_t = ::NS(buffer_size_t);

        bool success = (
            ( job.ptrContext() != nullptr ) &&
            ( job.context().hasSelectedNode() ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
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
            success = ( ( job.numParticleSets() == size_t{ 1 } ) &&
                ( job.particleSetIndicesBegin() != nullptr ) &&
                ( job.particleSetIndicesEnd()   != nullptr ) &&
                ( *( job.particleSetIndicesBegin() ) == size_t{ 0 } ) &&
                ( job.particleSetIndicesEnd() != job.particleSetIndicesBegin() ) &&
                ( job.particleSetIndex( size_t{ 0 } ) == size_t{ 0 } ) );
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
                ( job.ptrParticlesBuffer() == &pbuffer ) &&
                ( job.ptrBeamElementsBuffer() == &be_buffer ) );
        }

        return success;
    }

    bool test2_CreateTrackJobElemByElemOutputDelete(
        SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
        const SIXTRL_CXX_NAMESPACE::Buffer *const
            SIXTRL_RESTRICT ext_output_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const
            target_num_elem_by_elem_turns )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        bool success = false;

        using buffer_t            = st::Buffer;
        using size_t              = buffer_t::size_type;
        using particles_t         = st::Particles;
        using track_job_t         = st::TrackJobCl;
        using elem_by_elem_conf_t = track_job_t::elem_by_elem_config_t;

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        SIXTRL_ASSERT( target_num_elem_by_elem_turns > size_t{ 0 } );

        particles_t const* particles = particles_t::FromBuffer(
            pbuffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS = be_buffer.getNumObjects();
        size_t const NUM_PARTICLES = particles->getNumParticles();

        success = (
            ( NUM_BEAM_ELEMENTS > size_t{ 0 } ) &&
            ( NUM_PARTICLES > size_t{ 0 } ) &&
            ( job.ptrContext() != nullptr ) &&
            ( job.context().hasSelectedNode() ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
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
                ( job.ptrParticlesBuffer() == &pbuffer ) &&
                ( job.ptrBeamElementsBuffer() == &be_buffer ) );
        }

        return success;
    }

   bool test3_CreateTrackJobFullOutput(
        SIXTRL_CXX_NAMESPACE::TrackJobCl const& SIXTRL_RESTRICT_REF job,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF pbuffer,
        SIXTRL_CXX_NAMESPACE::Buffer const& SIXTRL_RESTRICT_REF be_buffer,
        const SIXTRL_CXX_NAMESPACE::Buffer *const
            SIXTRL_RESTRICT ext_output_buffer,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const num_beam_monitors,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type
            const target_num_output_turns,
        SIXTRL_CXX_NAMESPACE::Buffer::size_type const
            target_num_elem_by_elem_turns )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        bool success = false;

        using buffer_t    = st::Buffer;
        using size_t      = buffer_t::size_type;
        using particles_t = st::Particles;
        using track_job_t = st::TrackJobCl;
        using elem_by_elem_conf_t = track_job_t::elem_by_elem_config_t;

        buffer_t const* output_buffer = nullptr;
        elem_by_elem_conf_t const* elem_by_elem_conf = nullptr;

        particles_t const* particles = particles_t::FromBuffer(
            pbuffer, size_t{ 0 } );

        SIXTRL_ASSERT( particles != nullptr );

        size_t const NUM_BEAM_ELEMENTS = be_buffer.getNumObjects();
        size_t const NUM_PARTICLES = particles->getNumParticles();

        success = (
            ( job.ptrContext() != nullptr ) &&
            ( job.context().hasSelectedNode() ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
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
                        ::NS(Object_get_type_id)( be_buffer[ *be_mon_idx_it ] ) )
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
                ( job.ptrParticlesBuffer() == &pbuffer ) &&
                ( job.ptrBeamElementsBuffer() == &be_buffer ) );
        }

        return success;
    }
}
}

/* end: tests/sixtracklib/opencl/test_track_job_cl_opencl_cxx.cpp */
