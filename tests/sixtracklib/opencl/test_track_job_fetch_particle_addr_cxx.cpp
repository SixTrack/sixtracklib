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
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/particles/particles_addr.hpp"

TEST( CXXOpenCLTrackTrackJobFetchParticleAddr, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t           = st::Particles;
    using particles_addr_t      = ::NS(ParticlesAddr);
    using track_job_t           = st::TrackJobCl;
    using buffer_t              = track_job_t::buffer_t;
    using st_size_t             = track_job_t::size_type;
    using st_status_t           = track_job_t::status_t;
    using controller_t          = track_job_t::context_t;
    using node_id_t             = controller_t::node_id_t;
    using node_info_t           = controller_t::node_info_t;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t pb( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    st_size_t const NUM_PARTICLE_SETS = pb.getNumObjects();
    st_size_t const NUM_AVAILABLE_NODES = controller_t::NUM_AVAILABLE_NODES();

    if( NUM_AVAILABLE_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes available -> skipping tests\r\n";
        return;
    }

    std::vector< node_id_t > AVAILABLE_NODES(
        NUM_AVAILABLE_NODES, node_id_t{} );

    st_size_t const NUM_NODES = controller_t::GET_AVAILABLE_NODES(
        AVAILABLE_NODES.data(), AVAILABLE_NODES.size(), st_size_t{ 0 } );

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

        track_job_t track_job( tmp_device_id_str, pb, eb );
        controller_t* controller = track_job.ptrContext();

        ASSERT_TRUE( controller != nullptr );
        ASSERT_TRUE( controller->hasSelectedNode() );

        node_info_t const* selected_node_info =
            controller->ptrSelectedNodeInfo();

        node_id_t const default_node_id = controller->defaultNodeId();

        ASSERT_TRUE( selected_node_info != nullptr );
        std::cout << "[          ] [ INFO ] Selected Node \r\n";
        ::NS(ComputeNodeInfo_print_out)( selected_node_info, &default_node_id );
        std::cout << "\r\n";

        st_status_t status = st::ARCH_STATUS_SUCCESS;
        ASSERT_TRUE( track_job.canFetchParticleAddresses() );

        if( !track_job.hasParticleAddresses() )
        {
            status = track_job.fetchParticleAddresses();
        }

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.hasParticleAddresses() );
        ASSERT_TRUE( track_job.ptrParticleAddressesBuffer() != nullptr );

        st_size_t const slot_size =
            track_job.ptrParticleAddressesBuffer()->getSlotSize();

        for( st_size_t ii = st_size_t{ 0 } ; ii < NUM_PARTICLE_SETS ; ++ii )
        {
            particles_t const* cmp_particles = pb.get< particles_t >( ii );
            ASSERT_TRUE( cmp_particles != nullptr );

            particles_addr_t const* particles_addr =
                track_job.particleAddresses( ii );

            ASSERT_TRUE( particles_addr != nullptr );
            ASSERT_TRUE( cmp_particles->getNumParticles() ==
                         particles_addr->num_particles );

            bool const are_consistent =
                ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                    particles_addr, cmp_particles->getCApiPtr(), slot_size );

            ASSERT_TRUE( are_consistent );
        }

        status = track_job.clearAllParticleAddresses();
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.canFetchParticleAddresses() );
        ASSERT_TRUE( !track_job.hasParticleAddresses() );
    }
}

/* end: tests/sixtracklib/cuda/track/test_track_job_fetch_particle_addr_cxx.cpp */
