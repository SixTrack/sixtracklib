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
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/track_job_cl.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool test1_CreateTrackJobNoOutputDelete(
            const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer );

        bool test2_CreateTrackJobElemByElemOutputDelete(
            const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
            ::NS(buffer_size_t) const target_num_elem_by_elem_turns );

        bool test3_CreateTrackJobFullOutput(
            const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            const ::NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
            ::NS(buffer_size_t) const num_beam_monitors,
            ::NS(buffer_size_t) const target_num_output_turns,
            ::NS(buffer_size_t) const target_num_elem_by_elem_turns );
    }
}

TEST( C99_TrackJobClTests, CreateTrackJobNoOutputDelete )
{
    using track_job_t      = ::NS(TrackJobCl);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using particles_t      = ::NS(Particles);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

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

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* ===================================================================== *
     * First set of tests:
     * No Beam Monitors
     * No Elem by Elem config
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );

        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_create)( device_id_str );
        ASSERT_TRUE( job != nullptr );

        ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) ==
                     ::NS(TRACK_JOB_CL_ID) );

        ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                              ::NS(TRACK_JOB_CL_STR) ) == 0 );

        bool success = ::NS(TrackJobCl_reset_with_output)(
            job, pb, eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( success );
        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job, pb, eb, nullptr ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        job = ::NS(TrackJobCl_new_with_output)(
            device_id_str, pb, eb, nullptr, size_t{ 0 } );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) ==
                     ::NS(TRACK_JOB_CL_ID) );

        ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                                  ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job, pb, eb, nullptr ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) ==
                     ::NS(TRACK_JOB_CL_ID) );

        ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                                  ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job, pb, eb, nullptr ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 3 },
            &wrong_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) ==
                     ::NS(TRACK_JOB_CL_ID) );

        ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                                  ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job, pb, eb, nullptr ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, my_output_buffer, size_t{ 0 },
                nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_get_type_id)( job ) ==
                     ::NS(TRACK_JOB_CL_ID) );

        ASSERT_TRUE( std::strcmp( ::NS(TrackJob_get_type_str)( job ),
                                  ::NS(TRACK_JOB_CL_STR) ) == 0 );

        ASSERT_TRUE( st_test::test1_CreateTrackJobNoOutputDelete(
            job, pb, eb, my_output_buffer ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ================================================================= */
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";

    }

    std::cout << std::endl;

    ::NS(ClContext_delete)( context );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
    ::NS(Buffer_delete)( my_output_buffer );
}

TEST( C99_TrackJobClTests, CreateTrackJobElemByElemOutputDelete )
{
    using track_job_t      = ::NS(TrackJobCl);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using particles_t      = ::NS(Particles);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

    namespace st_test         = SIXTRL_CXX_NAMESPACE::tests;

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

    size_t const NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );

    size_t const NUM_PARTICLES =
        ::NS(Particles_get_num_of_particles)( particles );

    size_t const DUMP_ELEM_BY_ELEM_TURNS = size_t{  5u };

    ASSERT_TRUE( NUM_PARTICLES > size_t{ 0 } );
    ASSERT_TRUE( NUM_BEAM_ELEMENTS > size_t{ 0 } );

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* ===================================================================== *
     * Second set of tests:
     * No Beam Monitors
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );
        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_create)( device_id_str );
        ASSERT_TRUE( job != nullptr );

        bool success = ::NS(TrackJobCl_reset_with_output)(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( success );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        job = ::NS(TrackJobCl_new_with_output)(
            device_id_str, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, nullptr,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 3 },
            &wrong_particle_sets[ 0 ], eb, nullptr,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        ::NS(Buffer)*  my_output_buffer = ::NS(Buffer_new)( 0u );
        SIXTRL_ASSERT( my_output_buffer != nullptr );

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, my_output_buffer,
                DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test2_CreateTrackJobElemByElemOutputDelete(
            job, pb, eb, my_output_buffer, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(Buffer_delete)( my_output_buffer );
        ::NS(TrackJobCl_delete)( job );
        job = nullptr;
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";

    }

    std::cout << std::endl;

    /* ===================================================================== */

    ::NS(ClContext_delete)( context );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
}

TEST( C99_TrackJobClTests, CreateTrackJobBeamMonitorOutputDelete )
{
    using track_job_t      = ::NS(TrackJobCl);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using particles_t      = ::NS(Particles);
    using be_monitor_t     = ::NS(BeamMonitor);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

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

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* ===================================================================== *
     * Third set of tests:
     * Two Beam Monitors at end of lattice
     * No Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );
        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_create)( device_id_str );
        ASSERT_TRUE( job != nullptr );

        bool success = ::NS(TrackJobCl_reset)( job, pb, eb, nullptr );

        ASSERT_TRUE( success );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        job = ::NS(TrackJobCl_new)( device_id_str, pb, eb );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 3 },
            &wrong_particle_sets[ 0 ], eb, nullptr, size_t{ 0 }, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS, size_t{ 0 } ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        ::NS(Buffer)*  my_output_buffer = ::NS(Buffer_new)( 0u );
        SIXTRL_ASSERT( my_output_buffer != nullptr );

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, my_output_buffer, size_t{ 0 },
                nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, my_output_buffer, NUM_BEAM_MONITORS,
            NUM_TURNS, size_t{ 0 } ) );

        ::NS(Buffer_delete)( my_output_buffer );
        my_output_buffer = nullptr;

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";

    }

    std::cout << std::endl;

    /* ===================================================================== */

    ::NS(ClContext_delete)( context );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
}

TEST( C99_TrackJobClTests, CreateTrackJobFullDelete )
{
    using track_job_t      = ::NS(TrackJobCl);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using particles_t      = ::NS(Particles);
    using be_monitor_t     = ::NS(BeamMonitor);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

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

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* ===================================================================== *
     * Fourth set of tests:
     * Two Beam Monitors at end of lattice
     * Elem by Elem config
     * Output Buffer has to be present
     * --------------------------------------------------------------------- */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";

        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );
        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_create)( device_id_str );
        ASSERT_TRUE( job != nullptr );

        bool success = ::NS(TrackJobCl_reset_with_output)(
            job, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( success );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        job = ::NS(TrackJobCl_new_with_output)(
            device_id_str, pb, eb, nullptr, DUMP_ELEM_BY_ELEM_TURNS );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const good_particle_sets[] = { size_t{ 0 } };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, nullptr,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
            DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        size_t const wrong_particle_sets[] =
        {
            size_t{ 0 }, size_t{ 1 }, size_t{ 2 }
        };

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 3 },
            &wrong_particle_sets[ 0 ], eb, nullptr,
            DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, nullptr, NUM_BEAM_MONITORS, NUM_TURNS,
                DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;

        /* ----------------------------------------------------------------- */

        ::NS(Buffer)*  my_output_buffer = ::NS(Buffer_new)( 0u );
        SIXTRL_ASSERT( my_output_buffer != nullptr );

        job = ::NS(TrackJobCl_new_detailed)( device_id_str, pb, size_t{ 1 },
            &good_particle_sets[ 0 ], eb, my_output_buffer,
                DUMP_ELEM_BY_ELEM_TURNS, nullptr );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( st_test::test3_CreateTrackJobFullOutput(
            job, pb, eb, my_output_buffer, NUM_BEAM_MONITORS,
            NUM_TURNS, DUMP_ELEM_BY_ELEM_TURNS ) );

        ::NS(Buffer_delete)( my_output_buffer );
        my_output_buffer = nullptr;

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";

    }

    std::cout << std::endl;

    /* ===================================================================== */

    ::NS(ClContext_delete)( context );

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( in_particle_buffer );
}


TEST( C99_TrackJobClTests, CreateTrackJobTrackLineCompare )
{
    using buffer_t         = ::NS(Buffer)*;
    using particles_t      = ::NS(Particles)*;
    using buf_size_t       = ::NS(buffer_size_t);
    using track_job_t      = ::NS(TrackJobCl);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

    buffer_t pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
    particles_t cmp_particles = ::NS(Particles_add_copy)( cmp_track_pb,
        ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    buf_size_t const until_turn = 10;
    int status = ::NS(Track_all_particles_until_turn)(
        cmp_particles, eb, until_turn );

    SIXTRL_ASSERT( status == 0 );

    buf_size_t const num_beam_elements = ::NS(Buffer_get_num_of_objects)( eb );
    buf_size_t const num_lattice_parts = buf_size_t{ 10 };
    buf_size_t const num_elem_per_part = num_beam_elements / num_lattice_parts;

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        ::NS(Buffer)* track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
        particles_t particles = ::NS(Particles_add_copy)( track_pb,
            ::NS(Particles_buffer_get_const_particles)( pb, 0 ) );

        SIXTRL_ASSERT( particles != nullptr );
        ::NS(BeamMonitor_clear_all)( eb );

        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";
        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );
        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_new)( device_id_str, track_pb, eb );
        ASSERT_TRUE( job != nullptr );

        for( buf_size_t ii = buf_size_t{ 0 } ; ii < until_turn ; ++ii )
        {
            buf_size_t jj = buf_size_t{ 0 };

            for( ; jj < num_lattice_parts ; ++jj )
            {
                bool const is_last_in_turn = ( jj == ( num_lattice_parts - 1 ) );
                buf_size_t const begin_idx =  jj * num_elem_per_part;
                buf_size_t const end_idx   = ( !is_last_in_turn ) ?
                    begin_idx + num_elem_per_part : num_beam_elements;

                status = ::NS(TrackJobCl_track_line)(
                    job, begin_idx, end_idx, is_last_in_turn );

                 ASSERT_TRUE( status == 0 );
            }
        }

        ::NS(TrackJobCl_collect)( job );

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


        ::NS(TrackJobCl_delete)( job );
        ::NS(Buffer_delete)( track_pb );
    }

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( cmp_track_pb );
}

TEST( C99_TrackJobClTests, TrackParticles )
{
    using track_job_t      = ::NS(TrackJobCl);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using particles_t      = ::NS(Particles);
    using be_monitor_t     = ::NS(BeamMonitor);
    using part_index_t     = ::NS(particle_index_t);
    using cl_context_t     = ::NS(ClContext);
    using node_info_t      = ::NS(context_node_info_t);
    using node_id_t        = ::NS(context_node_id_t);
    using node_info_iter_t = node_info_t const*;

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

    size_t NUM_PARTICLES = ::NS(Particles_get_num_of_particles)( particles );
    size_t NUM_BEAM_ELEMENTS = ::NS(Buffer_get_num_of_objects)( eb );
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
        be_monitor_t* turn_by_turn_monitor = ::NS(BeamMonitor_new)( eb );
        ASSERT_TRUE( turn_by_turn_monitor != nullptr );

        ::NS(BeamMonitor_set_is_rolling)( turn_by_turn_monitor, false );
        ::NS(BeamMonitor_set_start)( turn_by_turn_monitor, UNTIL_TURN_ELEM_BY_ELEM );
        ::NS(BeamMonitor_set_num_stores)( turn_by_turn_monitor,
            UNTIL_TURN_TURN_BY_TURN - UNTIL_TURN_ELEM_BY_ELEM );
        ::NS(BeamMonitor_set_skip)( turn_by_turn_monitor, size_t{ 1 } );

        ++NUM_BEAM_MONITORS;
    }

    if( UNTIL_TURN > UNTIL_TURN_TURN_BY_TURN )
    {
        be_monitor_t* eot_monitor = ::NS(BeamMonitor_new)( eb );
        ASSERT_TRUE( eot_monitor != nullptr );

        ::NS(BeamMonitor_set_is_rolling)( eot_monitor, false );
        ::NS(BeamMonitor_set_start)( eot_monitor, UNTIL_TURN_TURN_BY_TURN );
        ::NS(BeamMonitor_set_num_stores)( eot_monitor,
            ( UNTIL_TURN - UNTIL_TURN_TURN_BY_TURN ) / SKIP_TURNS );
        ::NS(BeamMonitor_set_skip)( eot_monitor, SKIP_TURNS );

        ++NUM_BEAM_MONITORS;
    }

    /* -------------------------------------------------------------------- */
    /* create cmp particle and output data to verify track job performance  */

    size_t elem_by_elem_offset = size_t{ 0 };
    size_t beam_monitor_offset = size_t{ 0 };
    part_index_t min_turn_id   = part_index_t{ 0 };

    int ret = ::NS(OutputBuffer_prepare)( eb, cmp_output_buffer, cmp_particles,
        UNTIL_TURN_ELEM_BY_ELEM, &elem_by_elem_offset, &beam_monitor_offset,
            &min_turn_id );

    SIXTRL_ASSERT( ret == 0 );

    if( NUM_BEAM_MONITORS > size_t{ 0u } )
    {
        ret = ::NS(BeamMonitor_assign_output_buffer_from_offset)(
            eb, cmp_output_buffer, min_turn_id, beam_monitor_offset );

        ASSERT_TRUE( ret == 0 );
    }

    SIXTRL_ASSERT( ret == 0 );

    if( UNTIL_TURN_ELEM_BY_ELEM > size_t{ 0 } )
    {
        ::NS(ElemByElemConfig) config;
        ::NS(ElemByElemConfig_preset)( &config );

        ret = ::NS(ElemByElemConfig_init)( &config,
            cmp_particles, eb, part_index_t{ 0 }, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );

        ret = ::NS(ElemByElemConfig_assign_output_buffer)(
            &config, cmp_output_buffer, elem_by_elem_offset );

        ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );

        ret = ::NS(Track_all_particles_element_by_element_until_turn)(
            cmp_particles, &config, eb, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( ret == 0 );
    }

    if( UNTIL_TURN > UNTIL_TURN_ELEM_BY_ELEM )
    {
        ret = ::NS(Track_all_particles_until_turn)(
            cmp_particles, eb, UNTIL_TURN );

        SIXTRL_ASSERT( ret == 0 );
    }

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t* context  = ::NS(ClContext_create)();
    SIXTRL_ASSERT( context != nullptr );

    size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    /* -------------------------------------------------------------------- */
    /* perform tracking using a track_job: */

    node_info_iter_t node_it  =
        ::NS(ClContextBase_get_available_nodes_info_begin)( context );

    node_info_iter_t node_end =
        ::NS(ClContextBase_get_available_nodes_info_end)( context );

    node_id_t default_node_id =
        ::NS(ClContextBase_get_default_node_id)( context );

    for( size_t kk = size_t{ 0 } ; node_it != node_end ; ++node_it, ++kk )
    {
        ::NS(BeamMonitor_clear_all)( eb );
        ::NS(Particles_copy)( particles, ::NS(Particles_buffer_get_particles)(
            in_particle_buffer, size_t{ 0 } ) );

        std::cout << "node " << ( kk + size_t{ 1 } )
                  << " / " << num_nodes << "\r\n";
        node_id_t const node_id = ::NS(ComputeNodeInfo_get_id)( node_it );
        char device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, device_id_str, 16u ) );

        ::NS(ComputeNodeInfo_print_out)( node_it, &default_node_id );

        track_job_t* job = ::NS(TrackJobCl_new_with_output)(
            device_id_str, pb, eb, nullptr, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_has_output_buffer)( job ) );
        ASSERT_TRUE( ::NS(TrackJob_owns_output_buffer)( job ) );
        ASSERT_TRUE( ::NS(TrackJob_get_const_output_buffer)( job ) != nullptr );
        ASSERT_TRUE( ::NS(TrackJob_has_beam_monitor_output)( job ) ==
                     ( NUM_BEAM_MONITORS > size_t{ 0 } ) );

        ASSERT_TRUE( ::NS(TrackJob_get_num_beam_monitors)( job ) ==
                     NUM_BEAM_MONITORS );

        ASSERT_TRUE( ::NS(TrackJob_has_elem_by_elem_output)( job ) );
        ASSERT_TRUE( ::NS(TrackJob_has_elem_by_elem_config)( job ) );
        ASSERT_TRUE( ::NS(TrackJob_get_elem_by_elem_config)( job ) != nullptr );

        if( UNTIL_TURN_ELEM_BY_ELEM > size_t{ 0 } )
        {
            ret = ::NS(TrackJobCl_track_elem_by_elem)(
                job, UNTIL_TURN_ELEM_BY_ELEM );

            ASSERT_TRUE( ret == 0 );
        }

        if( UNTIL_TURN > UNTIL_TURN_ELEM_BY_ELEM )
        {
            ret = ::NS(TrackJobCl_track_until_turn)( job, UNTIL_TURN );
            ASSERT_TRUE( ret == 0 );
        }

        ::NS(TrackJobCl_collect)( job );

        /* --------------------------------------------------------------------- */
        /* compare */

        buffer_t const* ptr_output_buffer =
            ::NS(TrackJob_get_const_output_buffer)( job );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ==
                     ::NS(Buffer_get_num_of_objects)( cmp_output_buffer ) );

        double const ABS_ERR = double{ 2e-14 };

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

                    size_t const mm =
                        ::NS(Particles_get_num_of_particles)( cmp );

                    ASSERT_TRUE( mm > 0 );
                    ASSERT_TRUE( mm == static_cast< size_t >(
                        ::NS(Particles_get_num_of_particles)(
                            trk_particles ) ) );

                    if( 0 != ::NS(Particles_compare_values_with_treshold)(
                            cmp, trk_particles, ABS_ERR ) )
                    {
                        size_t const diff_index =
                                ::NS(Buffer_get_num_of_objects)( diff_buffer );

                        particles_t* diff = ::NS(Particles_new)(
                            diff_buffer, mm );

                        std::vector< ::NS(buffer_size_t) > max_diff_indices(
                            30, ::NS(buffer_size_t){ 0 } );

                        SIXTRL_ASSERT( diff != nullptr );

                        ::NS(Particles_calculate_difference)(
                            cmp, trk_particles, diff );

                        particles_t* max_diff = ::NS(Particles_new)(
                            diff_buffer, mm );

                        SIXTRL_ASSERT( max_diff != nullptr );

                        ::NS(Particles_get_max_difference)( max_diff,
                            max_diff_indices.data(), cmp, trk_particles );

                        std::cout <<
                                "-------------------------------------------"
                                "-------------------------------------------"
                                "-------------------------------------------"
                                "\r\nparticle set ii = " << ii <<
                                "\r\nmax_diff : \r\n";

                        ::NS(Particles_print_max_diff_out)(
                            max_diff, max_diff_indices.data() );

                        diff = ::NS(Particles_buffer_get_particles)(
                            diff_buffer, diff_index );


                        for( size_t jj = size_t{ 0 } ; jj < mm ; ++jj )
                        {
                            std::cout << "\r\nparticle index ii = "
                                      << jj << "\r\ncmp: \r\n";

                            ::NS(Particles_print_out_single)( cmp, jj );

                            std::cout << "\r\ntrk_particles: \r\n";
                            ::NS(Particles_print_out_single)(
                                trk_particles, jj );

                            std::cout << "\r\ndiff: \r\n";
                            ::NS(Particles_print_out_single)( diff, jj );
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

                ::NS(Buffer_delete)( diff_buffer );
                diff_buffer = nullptr;
            }
        }

        ASSERT_TRUE( ( ::NS(Particles_buffers_compare_values)(
                            ptr_output_buffer, cmp_output_buffer ) == 0 ) ||
            ( ::NS(Particles_buffers_compare_values_with_treshold)(
                ptr_output_buffer, cmp_output_buffer, ABS_ERR ) == 0 ) );

        ::NS(TrackJobCl_delete)( job );
        job = nullptr;
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "Skipping testcase because no OpenCL nodes are available";

    }

    std::cout << std::endl;

    /* --------------------------------------------------------------------- */

    ::NS(ClContext_delete)( context );

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
         const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
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
            ::NS(ClArgument) const* particle_buffer_arg =
                ::NS(TrackJobCl_get_const_particles_buffer_arg)( job );

            ::NS(ClArgument) const* beam_elements_buffer_arg =
                ::NS(TrackJobCl_get_const_beam_elements_buffer_arg)( job );

            success = (
                ( ::NS(TrackJobCl_get_const_context)( job ) != nullptr ) &&
                ( ::NS(ClContextBase_has_selected_node)(
                    ::NS(TrackJobCl_get_const_context)( job ) ) ) &&
                ( particle_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( particle_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                    particle_buffer_arg ) ==
                        ::NS(TrackJob_get_const_particles_buffer)( job ) ) &&
                ( beam_elements_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( beam_elements_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                        beam_elements_buffer_arg ) ==
                  ::NS(TrackJob_get_const_beam_elements_buffer)( job ) ) );
        }

        if( success )
        {
            ::NS(ClArgument) const* output_buffer_arg =
                    ::NS(TrackJobCl_get_const_output_buffer_arg)( job );

            if( ext_output_buffer != nullptr )
            {
                success = (
                    ( output_buffer_arg != SIXTRL_NULLPTR ) &&
                    ( ::NS(ClArgument_uses_cobj_buffer)( output_buffer_arg ) ) &&
                    (  ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                        output_buffer_arg ) ==
                       ::NS(TrackJob_get_const_output_buffer)( job ) ) &&
                    (  ::NS(TrackJob_get_const_output_buffer)(
                        job ) == ext_output_buffer ) &&
                    (  ::NS(TrackJob_has_output_buffer)( job ) ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) );
            }
            else
            {
                success = (
                    ( output_buffer_arg == nullptr ) &&
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
            const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
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
            ::NS(ClArgument) const* particle_buffer_arg =
                ::NS(TrackJobCl_get_const_particles_buffer_arg)( job );

            ::NS(ClArgument) const* beam_elements_buffer_arg =
                ::NS(TrackJobCl_get_const_beam_elements_buffer_arg)( job );

            success = (
                ( ::NS(TrackJobCl_get_const_context)( job ) != nullptr ) &&
                ( ::NS(ClContextBase_has_selected_node)(
                    ::NS(TrackJobCl_get_const_context)( job ) ) ) &&
                ( particle_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( particle_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                    particle_buffer_arg ) ==
                        ::NS(TrackJob_get_const_particles_buffer)( job ) ) &&
                ( beam_elements_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( beam_elements_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                        beam_elements_buffer_arg ) ==
                  ::NS(TrackJob_get_const_beam_elements_buffer)( job ) ) );
        }

        if( success )
        {
            ::NS(ClArgument) const* output_buffer_arg =
                    ::NS(TrackJobCl_get_const_output_buffer_arg)( job );

            success = (
                ( ::NS(TrackJob_has_output_buffer)( job ) ) &&
                ( output_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( output_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                    output_buffer_arg ) ==
                  ::NS(TrackJob_get_const_output_buffer)( job ) ) &&
                ( ( ( ext_output_buffer != nullptr ) &&
                    (  ::NS(TrackJob_get_const_output_buffer)( job ) ==
                        ext_output_buffer ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) ) ||
                  ( ( ext_output_buffer == nullptr ) &&
                    (  ::NS(TrackJob_owns_output_buffer)( job ) ) ) ) );
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
        const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
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
            ::NS(ClArgument) const* particle_buffer_arg =
                ::NS(TrackJobCl_get_const_particles_buffer_arg)( job );

            ::NS(ClArgument) const* beam_elements_buffer_arg =
                ::NS(TrackJobCl_get_const_beam_elements_buffer_arg)( job );

            success = (
                ( ::NS(TrackJobCl_get_const_context)( job ) != nullptr ) &&
                ( ::NS(ClContextBase_has_selected_node)(
                    ::NS(TrackJobCl_get_const_context)( job ) ) ) &&
                ( particle_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( particle_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                    particle_buffer_arg ) ==
                        ::NS(TrackJob_get_const_particles_buffer)( job ) ) &&
                ( beam_elements_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( beam_elements_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                        beam_elements_buffer_arg ) ==
                  ::NS(TrackJob_get_const_beam_elements_buffer)( job ) ) );
        }

        if( success )
        {
            ::NS(ClArgument) const* output_buffer_arg =
                    ::NS(TrackJobCl_get_const_output_buffer_arg)( job );

            success = (
                ( ::NS(TrackJob_has_output_buffer)( job ) ) &&
                ( output_buffer_arg != nullptr ) &&
                ( ::NS(ClArgument_uses_cobj_buffer)( output_buffer_arg ) ) &&
                ( ::NS(ClArgument_get_const_ptr_cobj_buffer)(
                    output_buffer_arg ) ==
                  ::NS(TrackJob_get_const_output_buffer)( job ) ) &&
                ( ( ( ext_output_buffer != nullptr ) &&
                    (  ::NS(TrackJob_get_const_output_buffer)( job ) ==
                        ext_output_buffer ) &&
                    ( !::NS(TrackJob_owns_output_buffer)( job ) ) ) ||
                  ( ( ext_output_buffer == nullptr ) &&
                    (  ::NS(TrackJob_owns_output_buffer)( job ) ) ) ) );
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

/* end: tests/sixtracklib/opencl/test_track_job_cl_opencl_c99.cpp */
