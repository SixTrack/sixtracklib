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

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/control/node_info.hpp"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"

TEST( CXX_CudaTrackJobTrackElemByElemTests,
      TrackElemByElemSingleParticleSetSimpleTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t           = st::Particles;
    using track_job_t           = st::CudaTrackJob;
    using node_index_t          = track_job_t::node_index_t;
    using buffer_t              = track_job_t::buffer_t;
    using buf_size_t            = track_job_t::size_type;
    using track_status_t        = track_job_t::track_status_t;
    using ctrl_status_t         = track_job_t::status_t;
    using node_id_t             = track_job_t::node_id_t;
    using real_t                = particles_t::real_t;
    using pindex_t              = particles_t::index_t;
    using elem_by_elem_config_t = ::NS(ElemByElemConfig);

    real_t const ABS_TOLERANCE = real_t{ 1e-14 };

    buffer_t in_particles( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t beam_elem_buffer( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t cmp_track_pb;

    buffer_t cmp_output_buffer;

    SIXTRL_ASSERT( in_particles.get< particles_t >(
        buf_size_t{ 0 } ) != nullptr );

    particles_t* cmp_particles = cmp_track_pb.addCopy(
        *in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t const NUM_PSETS  = buf_size_t{ 1 };
    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN_ELEM_BY_ELEM = buf_size_t{ 5 };

    buf_size_t num_elem_by_elem_objects = buf_size_t{ 0 };
    pindex_t const start_be_index = pindex_t{ 0 };

    pindex_t min_particle_id, max_particle_id;
    pindex_t min_at_element_id, max_at_element_id;
    pindex_t min_at_turn, max_at_turn;

    ctrl_status_t status = ::NS(OutputBuffer_get_min_max_attributes)(
        cmp_particles->getCApiPtr(), beam_elem_buffer.getCApiPtr(),
        &min_particle_id, &max_particle_id, &min_at_element_id,
        &max_at_element_id, &min_at_turn, &max_at_turn,
        &num_elem_by_elem_objects, start_be_index );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    buf_size_t elem_by_elem_out_offset_index = buf_size_t{ 0 };
    buf_size_t beam_element_out_offset_index = buf_size_t{ 0 };
    pindex_t   max_elem_by_elem_turn_id      = pindex_t{ -1 };

    status = ::NS(OutputBuffer_prepare_detailed)(
        beam_elem_buffer.getCApiPtr(), cmp_output_buffer.getCApiPtr(),
        min_particle_id, max_particle_id, min_at_element_id, max_at_element_id,
        min_at_turn, max_at_turn, UNTIL_TURN_ELEM_BY_ELEM,
        &elem_by_elem_out_offset_index, &beam_element_out_offset_index,
        &max_elem_by_elem_turn_id );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( max_elem_by_elem_turn_id >= pindex_t{ 0 } );

    elem_by_elem_config_t elem_by_elem_config;
    ::NS(ElemByElemConfig_preset)( &elem_by_elem_config );

    status = ::NS(ElemByElemConfig_init_detailed)( &elem_by_elem_config,
        ::NS(ELEM_BY_ELEM_ORDER_DEFAULT), min_particle_id, max_particle_id,
        min_at_element_id, max_at_element_id, min_at_turn,
        max_elem_by_elem_turn_id, true );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    status = ::NS(ElemByElemConfig_assign_output_buffer)(
        &elem_by_elem_config, cmp_output_buffer.getCApiPtr(),
            elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_elem_by_elem_until_turn_cpu)(
        cmp_track_pb.getCApiPtr(), NUM_PSETS, &track_pset_index,
            beam_elem_buffer.getCApiPtr(), &elem_by_elem_config,
                UNTIL_TURN_ELEM_BY_ELEM );

    SIXTRL_ASSERT( track_status == st::TRACK_SUCCESS );

    particles_t* cmp_elem_by_elem_particles =
        cmp_output_buffer.get< particles_t >( elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( cmp_elem_by_elem_particles != nullptr );

    /* -------------------------------------------------------------------- */
    /* Retrieve list of available nodes */

    buf_size_t const num_avail_nodes = st::CudaTrackJob::NumAvailableNodes();

    if( num_avail_nodes == buf_size_t{ 0 } )
    {
        std::cerr << "[          ] [ INFO ] \r\n"
                  << "[          ] [ INFO ] "
                  << "!!!!!!!! No cuda nodes found -> skipping test !!!!!!\r\n"
                  << "[          ] [ INFO ]" << std::endl;

        return;
    }

    std::vector< node_id_t > avail_node_ids( num_avail_nodes );

    buf_size_t const num_nodes = st::CudaTrackJob::GetAvailableNodeIdsList(
        avail_node_ids.size(), avail_node_ids.data() );

    ASSERT_TRUE( num_nodes == num_avail_nodes );

    for( auto const& node_id : avail_node_ids )
    {
        /* Create a dedicated buffer for tracking */
        buffer_t track_pb;
        particles_t* particles = track_pb.addCopy(
            *in_particles.get< particles_t >( buf_size_t{ 0 } ) );

        SIXTRL_ASSERT( particles != nullptr );

        /* Create a track job on the current node */
        track_job_t track_job( node_id.toString(),
            track_pb, beam_elem_buffer, nullptr, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( track_job.hasOutputBuffer() );
        ASSERT_TRUE( track_job.ownsOutputBuffer() );
        ASSERT_TRUE( track_job.hasElemByElemOutput() );
        ASSERT_TRUE( track_job.elemByElemOutputBufferOffset() ==
                     elem_by_elem_out_offset_index );

        ASSERT_TRUE( track_job.ptrCudaController() != nullptr );
        ASSERT_TRUE( track_job.hasSelectedNode() );
        ASSERT_TRUE( track_job.ptrCudaController()->isSelectedNode( node_id ) );

        auto node_info = track_job.ptrCudaController()->ptrNodeInfo( node_id );
        ASSERT_TRUE( node_info != nullptr );

        std::cout << "[          ] [ INFO ] Selected Node \r\n";
        node_info->printOut();

        if( !track_job.isInDebugMode() )
        {
            track_job.enableDebugMode();
        }

        ASSERT_TRUE( track_job.requiresCollecting() );
        ASSERT_TRUE( track_job.isInDebugMode() );

        track_status = track_job.trackElemByElem( UNTIL_TURN_ELEM_BY_ELEM );
        ASSERT_TRUE( track_status == st::TRACK_SUCCESS );

        status = track_job.collectParticles();
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        SIXTRL_ASSERT( track_job.ptrParticlesBuffer() == &track_pb );
        particles = track_pb.get< particles_t >( buf_size_t{ 0 } );

        ASSERT_TRUE(
            ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
            ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
              ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
                ( 0 == ::NS(Particles_compare_values_with_treshold)(
                        cmp_particles->getCApiPtr(), particles->getCApiPtr(),
                            ABS_TOLERANCE ) ) ) ) );

        status = track_job.collectOutput();
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.ptrOutputBuffer() != nullptr );

        particles_t* elem_by_elem_particles =
            track_job.ptrOutputBuffer()->get< particles_t >(
                elem_by_elem_out_offset_index );

        SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

        ASSERT_TRUE(
            ( cmp_elem_by_elem_particles != nullptr ) &&
            ( elem_by_elem_particles != nullptr ) &&
            ( ( 0 == ::NS(Particles_compare_values)(
                cmp_elem_by_elem_particles->getCApiPtr(),
                elem_by_elem_particles->getCApiPtr() ) ) ||
              ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
                ( 0 == ::NS(Particles_compare_values_with_treshold)(
                        cmp_elem_by_elem_particles->getCApiPtr(),
                        elem_by_elem_particles->getCApiPtr(),
                        ABS_TOLERANCE ) ) ) ) );

        /* ***************************************************************** */
        /* Switch to non-debug mode: */

        status = particles->copy(
            in_particles.get< particles_t >( buf_size_t{ 0 } ) );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        track_job.disableDebugMode();

        ASSERT_TRUE( !track_job.isInDebugMode() );

        status = track_job.reset( track_pb,
            beam_elem_buffer, nullptr, UNTIL_TURN_ELEM_BY_ELEM );

        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

        /* Check whether the update of the particles state has worked */
        status = track_job.collectParticles();
        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        particles = track_pb.get< particles_t >( buf_size_t{ 0 } );
        SIXTRL_ASSERT( particles != nullptr );
        SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
            particles->getCApiPtr(),
            in_particles.get< particles_t >( buf_size_t{ 0 } )->getCApiPtr() ) );

        /* Perform tracking again, this time not in debug mode */
        track_status = track_job.trackElemByElem( UNTIL_TURN_ELEM_BY_ELEM );
        ASSERT_TRUE( track_status == st::TRACK_SUCCESS );

        /* Collect the results again and ... */
        status = track_job.collectParticles();
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        SIXTRL_ASSERT( track_job.ptrParticlesBuffer() == &track_pb );
        particles = track_pb.get< particles_t >( buf_size_t{ 0 } );

        status = track_job.collectOutput();
        ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
        ASSERT_TRUE( track_job.ptrOutputBuffer() != nullptr );

        elem_by_elem_particles =
            track_job.ptrOutputBuffer()->get< particles_t >(
                elem_by_elem_out_offset_index );

        SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

        /* ... compare against the cpu tracking result */

        ASSERT_TRUE(
            ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
            ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
              ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
                ( 0 == ::NS(Particles_compare_values_with_treshold)(
                        cmp_particles->getCApiPtr(), particles->getCApiPtr(),
                            ABS_TOLERANCE ) ) ) ) );


        ASSERT_TRUE(
            ( cmp_elem_by_elem_particles != nullptr ) &&
            ( elem_by_elem_particles != nullptr ) &&
            ( ( 0 == ::NS(Particles_compare_values)(
                cmp_elem_by_elem_particles->getCApiPtr(),
                elem_by_elem_particles->getCApiPtr() ) ) ||
              ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
                ( 0 == ::NS(Particles_compare_values_with_treshold)(
                        cmp_elem_by_elem_particles->getCApiPtr(),
                        elem_by_elem_particles->getCApiPtr(),
                        ABS_TOLERANCE ) ) ) ) );
    }
}

/* end: tests/sixtracklib/cuda/track/test_track_job_track_elem_by_elem_cxx.cpp */
