#include "sixtracklib/cuda/track_job.h"

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

TEST( CXX_CudaTrackJobFetchParticleAddr, BasicUsage )
{
    using node_id_t        = ::NS(NodeId);
    using track_job_t      = ::NS(CudaTrackJob);
    using controller_t     = ::NS(CudaController);
    using node_info_t      = ::NS(CudaNodeInfo);
    using node_index_t     = ::NS(node_index_t);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using ctrl_status_t    = ::NS(arch_status_t);
    using particles_addr_t = ::NS(ParticlesAddr);
    using particles_t      = ::NS(Particles);
    using real_t           = ::NS(particle_real_t);
    using pindex_t         = ::NS(particle_index_t);

    buffer_t* eb = ::NS(Buffer_new_from_file)( 
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    
    buffer_t* pb = ::NS(Buffer_new_from_file)( 
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP ) );

    size_t const num_particle_sets = ::NS(Buffer_get_num_of_objects)( pb );
    size_t const num_avail_nodes = ::NS(CudaTrackJob_get_num_available_nodes)();

    if( num_avail_nodes == size_t{ 0 } )
    {
        std::cerr << "[          ] [ INFO ] \r\n"
                  << "[          ] [ INFO ] "
                  << "!!!!!!!! No cuda nodes found -> skipping test !!!!!!\r\n"
                  << "[          ] [ INFO ]" << std::endl;

        return;
    }

    std::vector< node_id_t > avail_node_ids( num_avail_nodes );

    size_t const num_nodes = ::NS(CudaTrackJob_get_available_node_ids_list)(
        avail_node_ids.size(), avail_node_ids.data() );

    ASSERT_TRUE( num_nodes == num_avail_nodes );

    for( auto const& node_id : avail_node_ids )
    {
        /* Create a track job on the current node */
        std::string const node_id_str = node_id.toString();
        
        track_job_t* track_job = ::NS(CudaTrackJob_new)( 
            node_id_str.c_str(), pb, eb );

        ASSERT_TRUE( track_job != nullptr );

        controller_t* ptr_ctrl =
            ::NS(CudaTrackJob_get_ptr_controller)( track_job );

        ASSERT_TRUE( ptr_ctrl != nullptr );
        ASSERT_TRUE( ::NS(Controller_has_selected_node)( ptr_ctrl ) );

        ASSERT_TRUE( ::NS(Controller_get_node_index_by_node_id)(
            ptr_ctrl, &node_id ) != ::NS(NODE_UNDEFINED_INDEX) );

        ASSERT_TRUE( ::NS(Controller_get_selected_node_index)( ptr_ctrl ) ==
                     ::NS(Controller_get_node_index_by_node_id)(
                         ptr_ctrl, &node_id ) );

        node_info_t const* node_info = ::NS(CudaController_get_ptr_node_info)(
            ptr_ctrl, node_id_str.c_str() );

        ASSERT_TRUE( node_info != nullptr );

        std::cout << "[          ] [ INFO ] Selected Node \r\n";
        ::NS(NodeInfo_print_out)( node_info );

        ctrl_status_t status = ::NS(ARCH_STATUS_SUCCESS);

        if( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) )
        {
            status = ::NS(TrackJobNew_enable_debug_mode)( track_job );
        }

        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        ASSERT_TRUE( ::NS(TrackJobNew_is_in_debug_mode)( track_job ) );
        
        ASSERT_TRUE( ::NS(TrackJobNew_can_fetch_particle_addresses)( 
            track_job ) );
        
        if( !::NS(TrackJobNew_has_particle_addresses)( track_job ) )
        {
            status = ::NS(TrackJobNew_fetch_particle_addresses)( track_job );
        }

        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        ASSERT_TRUE( ::NS(TrackJobNew_has_particle_addresses)( track_job ) );
        ASSERT_TRUE( ::NS(TrackJobNew_get_ptr_particle_addresses_buffer)(
            track_job ) != nullptr );        
        
        size_t const slot_size = ::NS(Buffer_get_slot_size)(
            ::NS(TrackJobNew_get_ptr_particle_addresses_buffer)( track_job ) );

        for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
        {
            particles_t const* cmp_particles = 
                ::NS(Particles_buffer_get_const_particles)( pb, ii );

            ASSERT_TRUE( cmp_particles != nullptr );

            particles_addr_t const* particles_addr =
                ::NS(TrackJobNew_get_particle_addresses)( track_job, ii );

            ASSERT_TRUE( particles_addr != nullptr );
            ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp_particles )
                         == particles_addr->num_particles );

            bool const are_consistent =
                ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                    particles_addr, cmp_particles, slot_size );
                
            ASSERT_TRUE( are_consistent );
        }
        
        status = ::NS(TrackJobNew_clear_all_particle_addresses)( track_job );        
        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        ASSERT_TRUE( ::NS(TrackJobNew_can_fetch_particle_addresses)( 
            track_job ) ); 
        
        ASSERT_TRUE( !::NS(TrackJobNew_has_particle_addresses)( track_job ) );
        
        status = ::NS(TrackJobNew_disable_debug_mode)( track_job );
        
        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        ASSERT_TRUE( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) );
        
        status = ::NS(TrackJobNew_fetch_particle_addresses)( track_job );
        
        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
        ASSERT_TRUE( ::NS(TrackJobNew_has_particle_addresses)( track_job ) );
        
        for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
        {
            particles_t const* cmp_particles = ::NS(Particles_buffer_get_const_particles)( pb, ii );

            ASSERT_TRUE( cmp_particles != nullptr );

            particles_addr_t const* particles_addr =
                ::NS(TrackJobNew_get_particle_addresses)( track_job, ii );

            ASSERT_TRUE( particles_addr != nullptr );
            ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp_particles )
                         == particles_addr->num_particles );

            bool const are_consistent =
                ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                    particles_addr, cmp_particles, slot_size );
                
            ASSERT_TRUE( are_consistent );
        }
        
        ::NS(TrackJobNew_delete)( track_job );
        track_job = nullptr;
    }
    
    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
}

/* end: tests/sixtracklib/cuda/track/test_track_job_fetch_particle_addr_c99.cpp */
