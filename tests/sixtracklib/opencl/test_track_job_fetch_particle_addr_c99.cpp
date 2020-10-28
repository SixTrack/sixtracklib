#include "sixtracklib/opencl/track_job_cl.h"

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
#include "sixtracklib/common/particles/particles_addr.h"

TEST( C99OpenCLTrackTrackJobFetchParticleAddr, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t           = ::NS(Particles);
    using particles_addr_t      = ::NS(ParticlesAddr);
    using track_job_t           = ::NS(TrackJobCl);
    using buffer_t              = ::NS(Buffer);
    using st_size_t             = ::NS(buffer_size_t);
    using st_status_t           = ::NS(arch_status_t);
    using controller_t          = ::NS(ClContext);
    using node_id_t             = ::NS(ComputeNodeId);
    using node_info_t           = ::NS(ComputeNodeInfo);

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( eb != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( pb != SIXTRL_NULLPTR );

    st_size_t const NUM_PARTICLE_SETS = ::NS(Buffer_get_num_of_objects)( pb );
    st_size_t const NUM_AVAILABLE_NODES = ::NS(OpenCL_num_available_nodes)(
        "SIXTRACKLIB_DEVICES" );

    if( NUM_AVAILABLE_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes available -> skipping tests\r\n";
        return;
    }

    std::vector< node_id_t > AVAILABLE_NODES(
        NUM_AVAILABLE_NODES, node_id_t{} );

    st_size_t const NUM_NODES = ::NS(OpenCL_get_available_nodes)(
        AVAILABLE_NODES.data(), AVAILABLE_NODES.size() );

    ASSERT_TRUE( NUM_NODES > size_t{ 0 } );

    for( auto const& node_id : AVAILABLE_NODES )
    {
        char tmp_device_id_str[] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
            &node_id, tmp_device_id_str, 16u ) );

        track_job_t* track_job = NS(TrackJobCl_new)(
            tmp_device_id_str, pb, eb );
        ASSERT_TRUE( track_job != nullptr );

        controller_t* controller = NS(TrackJobCl_get_context)( track_job );

        ASSERT_TRUE( controller != nullptr );
        ASSERT_TRUE( controller->hasSelectedNode() );

        node_info_t const* selected_node_info =
            ::NS(ClContextBase_get_selected_node_info)( controller );

        node_id_t const default_node_id =
            ::NS(ClContextBase_get_default_node_id)( controller );

        ASSERT_TRUE( selected_node_info != nullptr );
        std::cout << "[          ] [ INFO ] Selected Node \r\n";
        ::NS(ComputeNodeInfo_print_out)( selected_node_info, &default_node_id );
        std::cout << "\r\n";

        st_status_t status = st::ARCH_STATUS_SUCCESS;
        ASSERT_TRUE( ::NS(TrackJob_can_fetch_particles_addr)( track_job ) );

        if( !::NS(TrackJob_has_particles_addr)( track_job ) )
        {
            status = NS(TrackJob_fetch_particle_addresses)( track_job );
        }

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( ::NS(TrackJob_has_particles_addr)( track_job ) );
        ASSERT_TRUE( ::NS(TrackJob_get_const_particles_addr_buffer)(
                        track_job ) );

        st_size_t const slot_size = ::NS(Buffer_get_slot_size)(
            ::NS(TrackJob_get_const_particles_addr_buffer)( track_job ) );

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_PARTICLE_SETS ; ++ii )
        {
            particles_t const* cmp_particles =
                ::NS(Particles_buffer_get_const_particles)( pb, ii );
            ASSERT_TRUE( cmp_particles != nullptr );

            particles_addr_t const* particles_addr =
                ::NS(TrackJob_particle_addresses)( track_job, ii );

            ASSERT_TRUE( particles_addr != nullptr );
            ASSERT_TRUE( ::NS(Particles_get_num_of_particles)(
                cmp_particles ) == particles_addr->num_particles );

            bool const are_consistent =
                ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                    particles_addr, cmp_particles, slot_size );

            ASSERT_TRUE( are_consistent );
        }

        status = ::NS(TrackJob_clear_all_particle_addresses)( track_job );
        ASSERT_TRUE(  status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE(  ::NS(TrackJob_can_fetch_particles_addr)( track_job ) );
        ASSERT_TRUE( !::NS(TrackJob_has_particles_addr)( track_job ) );

        ::NS(TrackJob_delete)( track_job );
        track_job  = nullptr;
        controller = nullptr;
    }

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );

    pb = nullptr;
    eb = nullptr;
}

/* end: tests/sixtracklib/cuda/track/test_track_job_fetch_particle_addr_cxx.cpp */
