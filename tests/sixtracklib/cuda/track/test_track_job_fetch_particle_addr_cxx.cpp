#include "sixtracklib/cuda/track_job.hpp"

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

TEST( CXX_CudaTrackJobFetchParticleAddr, BasicUsage )
{
    namespace st           = SIXTRL_CXX_NAMESPACE;

    using track_job_t      = st::CudaTrackJob;
    using node_id_t        = track_job_t::node_id_t;
    using size_t           = track_job_t::size_type;
    using buffer_t         = track_job_t::buffer_t;
    using ctrl_status_t    = track_job_t::status_t;
    using particles_addr_t = track_job_t::particles_addr_t;
    using particles_t      = st::Particles;
    using real_t           = particles_t::real_t;
    using pindex_t         = particles_t::index_t;
    using address_t        = ::NS(buffer_addr_t);
    using addr_diff_t      = ::NS(buffer_addr_diff_t);

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t pb( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP ) );

    size_t const num_particle_sets = pb.getNumObjects();

    size_t const num_avail_nodes = st::CudaTrackJob::NumAvailableNodes();

    address_t const real_size = sizeof( real_t );
    address_t const idx_size  = sizeof( pindex_t );

    if( num_avail_nodes == size_t{ 0 } )
    {
        std::cerr << "[          ] [ INFO ] \r\n"
                  << "[          ] [ INFO ] "
                  << "!!!!!!!! No cuda nodes found -> skipping test !!!!!!\r\n"
                  << "[          ] [ INFO ]" << std::endl;

        return;
    }

    std::vector< node_id_t > avail_node_ids( num_avail_nodes );

    size_t const num_nodes = st::CudaTrackJob::GetAvailableNodeIdsList(
        avail_node_ids.size(), avail_node_ids.data() );

    ASSERT_TRUE( num_nodes == num_avail_nodes );

    for( auto const& node_id : avail_node_ids )
    {
        /* Create a track job on the current node */
        track_job_t track_job( node_id.toString(), pb, eb );

        ASSERT_TRUE( track_job.ptrCudaController() != nullptr );
        ASSERT_TRUE( track_job.hasSelectedNode() );
        ASSERT_TRUE( track_job.ptrCudaController()->isSelectedNode( node_id ) );

        auto node_info = track_job.ptrCudaController()->ptrNodeInfo( node_id );
        ASSERT_TRUE( node_info != nullptr );

        std::cout << "[          ] [ INFO ] Selected Node \r\n";
        node_info->printOut();

        ctrl_status_t status = st::ARCH_STATUS_SUCCESS;

        if( !track_job.isInDebugMode() )
        {
            status = track_job.enableDebugMode();
        }

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.isInDebugMode() );
        
        ASSERT_TRUE( track_job.canFetchParticleAddresses() );
        
        if( !track_job.hasParticleAddresses() )
        {
            status = track_job.fetchParticleAddresses();
        }

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.hasParticleAddresses() );
        ASSERT_TRUE( track_job.ptrParticleAddressesBuffer() != nullptr );        
        
        size_t const slot_size = 
            track_job.ptrParticleAddressesBuffer()->getSlotSize();

        for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
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
        
        status = track_job.disableDebugMode();
        
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( !track_job.isInDebugMode() );
        
        status = track_job.fetchParticleAddresses();
        
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.hasParticleAddresses() );
        
        for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
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
    }
}

/* end: tests/sixtracklib/cuda/track/test_track_job_fetch_particle_addr_cxx.cpp */
