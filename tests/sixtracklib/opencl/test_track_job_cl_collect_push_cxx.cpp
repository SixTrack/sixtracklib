#include "sixtracklib/opencl/track_job_cl.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/beam_elements.hpp"
#include "sixtracklib/common/particles.hpp"

TEST( CXX_TrackJobClCollectPushTests, TestCollectAndPush )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using track_job_t        = st::TrackJobCl;
    using buffer_t           = st::Buffer;
    using particles_t        = st::Particles;
    using cl_context_t       = st::ClContext;
    using drift_t            = st::Drift;
    using cavity_t           = st::Cavity;
    using be_monitor_t       = st::BeamMonitor;
    using limit_rect_t       = st::LimitRect;
    using limit_ellipse_t    = st::LimitEllipse;
    using node_info_t        = cl_context_t::node_info_t;
    using node_id_t          = cl_context_t::node_id_t;
    using node_info_iter_t   = node_info_t const*;
    using size_t             = buffer_t::size_type;
    using be_monitor_turn_t  = be_monitor_t::turn_t;
    using be_monitor_addr_t  = be_monitor_t::address_t;
    using particle_index_t   = particles_t::index_t;

    double const EPS = std::numeric_limits< double >::epsilon();

    /* Initialize pseudo random generator to init some parameters during the
       test */

    std::random_device rnd_device;
    std::mt19937_64 prng( rnd_device() );
    std::uniform_real_distribution< double > dbl_dist( 0.0, 1.0 );

    auto gen = [ &prng, &dbl_dist ]() { return dbl_dist( prng ); };

    /* Create particles buffers */

    buffer_t init_eb;
    buffer_t init_pb;

    size_t constexpr NUM_PARTICLES = size_t{ 100 };

    particles_t* particles = init_pb.createNew< particles_t >( NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );
    ::NS(Particles_random_init)( particles->getCApiPtr() );

    drift_t*  e01_drift = init_eb.add< drift_t  >( double{ 1.0 } );

    cavity_t* e02_cavity = init_eb.add< cavity_t >(
        double{ 100e3 }, double{ 400e6 }, double { 0.0 } );

    drift_t* e03_drift = init_eb.add< drift_t >( double{ 3.0 } );

    double const e04_min_x = double{ -1.0 };
    double const e04_max_x = double{ +1.0 };
    double const e04_min_y = double{ -2.0 };
    double const e04_max_y = double{  2.0 };

    limit_rect_t* e04_limit_rect = init_eb.add< limit_rect_t >(
        e04_min_x, e04_max_x, e04_min_y, e04_max_y );

    drift_t* e05_drift = init_eb.add< drift_t >( double{ 5.0 } );

    be_monitor_turn_t const e06_num_stores = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e06_start_turn = be_monitor_turn_t{ 0 };
    be_monitor_turn_t const e06_skip_turns = be_monitor_turn_t{ 0 };

    bool const e06_monitor_is_rolling = false;
    bool const e06_monitor_is_turn_ordered = false;
    be_monitor_addr_t const out_addr = be_monitor_addr_t{ 0 };

    be_monitor_t* e06_monitor = init_eb.add< be_monitor_t >(
        e06_num_stores, e06_start_turn, e06_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e06_monitor_is_rolling, e06_monitor_is_turn_ordered );

    drift_t* e07_drift = init_eb.add< drift_t >( double{ 7.0 } );

    double const e08_x_half_axis = double{ 0.5  };
    double const e08_y_half_axis = double{ 0.35 };

    limit_ellipse_t* e08_limit_ell = init_eb.add< limit_ellipse_t >(
        e08_x_half_axis, e08_y_half_axis );

    be_monitor_turn_t const e09_num_stores = be_monitor_turn_t{ 5  };
    be_monitor_turn_t const e09_start_turn = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e09_skip_turns = be_monitor_turn_t{ 5  };

    bool const e09_monitor_is_rolling = true;
    bool const e09_monitor_is_turn_ordered = false;

    be_monitor_t* e09_monitor = init_eb.add< be_monitor_t >(
        e09_num_stores, e09_start_turn, e09_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e09_monitor_is_rolling, e09_monitor_is_turn_ordered );

    SIXTRL_ASSERT( init_eb.getNumObjects() == size_t{ 9 } );

    /* ---------------------------------------------------------------------- */
    /* Prepare device index to device_id_str map */

    cl_context_t context;
    size_t const num_nodes = context.numAvailableNodes();

    /* ===================================================================== */

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

        /* ----------------------------------------------------------------- */
        /* Create per-device copies of the particle and beam elements buffer */

        buffer_t pb( init_pb );
        particles = pb.get< particles_t >( 0 );
        SIXTRL_ASSERT( particles != nullptr );

        buffer_t eb( init_eb );
        e01_drift      = eb.get< drift_t >( 0 );
        e02_cavity     = eb.get< cavity_t >( 1 );
        e03_drift      = eb.get< drift_t >( 2 );
        e04_limit_rect = eb.get< limit_rect_t >( 3 );
        e05_drift      = eb.get< drift_t >( 4 );
        e06_monitor    = eb.get< be_monitor_t >( 5 );
        e07_drift      = eb.get< drift_t >( 6 );
        e08_limit_ell  = eb.get< limit_ellipse_t >( 7 );
        e09_monitor    = eb.get< be_monitor_t >( 8 );

        ASSERT_TRUE( e01_drift      != nullptr );
        ASSERT_TRUE( e02_cavity     != nullptr );
        ASSERT_TRUE( e03_drift      != nullptr );
        ASSERT_TRUE( e04_limit_rect != nullptr );
        ASSERT_TRUE( e05_drift      != nullptr );
        ASSERT_TRUE( e06_monitor    != nullptr );
        ASSERT_TRUE( e07_drift      != nullptr );
        ASSERT_TRUE( e08_limit_ell  != nullptr );
        ASSERT_TRUE( e09_monitor    != nullptr );

        track_job_t  job( device_id_str, pb, eb );
        ASSERT_TRUE( job.requiresCollecting() );
        ASSERT_TRUE( job.hasOutputBuffer() );
        ASSERT_TRUE( job.hasBeamMonitorOutput() );
        ASSERT_TRUE( job.ptrParticlesBuffer() == &pb );
        ASSERT_TRUE( job.ptrBeamElementsBuffer() == &eb );
        ASSERT_TRUE( job.ptrOutputBuffer() != nullptr );

        buffer_t copy_of_eb( eb );
        SIXTRL_ASSERT( copy_of_eb.getNumObjects() == eb.getNumObjects() );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 0 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 0 ) != e01_drift );
        SIXTRL_ASSERT( copy_of_eb.get< cavity_t >( 1 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< cavity_t >( 1 ) != e02_cavity );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 2 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 2 ) != e03_drift );
        SIXTRL_ASSERT( copy_of_eb.get< limit_rect_t >( 3 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< limit_rect_t >( 3 ) != e04_limit_rect );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 4 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 4 ) != e05_drift );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 ) != e06_monitor );

        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 6 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 6 ) != e07_drift );
        SIXTRL_ASSERT( copy_of_eb.get< limit_ellipse_t >( 7 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< limit_ellipse_t >( 7 ) != e08_limit_ell );

        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 ) != e09_monitor );

        /* The beam monitors on the host side should have output adddr == 0: */
        SIXTRL_ASSERT( e06_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );
        SIXTRL_ASSERT( e09_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );

        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 )->getOutAddress() ==
                       e06_monitor->getOutAddress() );

        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 )->getOutAddress() ==
                       e09_monitor->getOutAddress() );

        buffer_t copy_of_pb( pb );
        SIXTRL_ASSERT( copy_of_pb.getNumObjects() == pb.getNumObjects() );
        SIXTRL_ASSERT( copy_of_pb.get< particles_t >( 0 ) != nullptr );
        SIXTRL_ASSERT( copy_of_pb.get< particles_t >( 0 ) != particles );

        buffer_t* ptr_output_buffer = job.ptrOutputBuffer();
        ASSERT_TRUE( ptr_output_buffer != nullptr );
        ASSERT_TRUE( ptr_output_buffer->getNumObjects() > size_t{ 0 } );
        ASSERT_TRUE( ptr_output_buffer->getNumObjects() ==
                     job.numBeamMonitors() );

        buffer_t copy_of_output_buffer( *ptr_output_buffer );
        SIXTRL_ASSERT( copy_of_output_buffer.getNumObjects() ==
                       ptr_output_buffer->getNumObjects() );

        /* ----------------------------------------------------------------- */
        /* Set some parameters on the host side to different values. These will
         * be overwritten with the original values when we collect */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */
        SIXTRL_ASSERT( ::NS(Particles_compare_values)( particles->getCApiPtr(),
            copy_of_pb.get< particles_t >( 0 )->getCApiPtr() ) == 0 );
        std::generate( particles->getX(), particles->getX() + NUM_PARTICLES, gen );
        SIXTRL_ASSERT( ::NS(Particles_compare_values)( particles->getCApiPtr(),
            copy_of_pb.get< particles_t >( 0 ) ) != 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */
        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e01_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 0 )->getCApiPtr() ) == 0 );
        e01_drift->setLength( double{0.5} * e01_drift->getLength() );
        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e01_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 0 )->getCApiPtr() ) != 0 );

        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity->getCApiPtr(),
            copy_of_eb.get< cavity_t >( 1 )->getCApiPtr() ) == 0 );
        e02_cavity->setVoltage( double{2.0} * e02_cavity->getVoltage() );
        e02_cavity->setFrequency( double{2.0} * e02_cavity->getFrequency() );
        e02_cavity->setLag( double{ 1.0 } );
        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity->getCApiPtr(),
            copy_of_eb.get< cavity_t >( 1 )->getCApiPtr() ) != 0 );

        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e03_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 2 )->getCApiPtr() ) == 0 );
        e03_drift->setLength( double{ 0.0 } );
        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e03_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 2 )->getCApiPtr() ) != 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output: */
        for( size_t ii = 0u ; ii < ptr_output_buffer->getNumObjects() ; ++ii )
        {
            SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
                ptr_output_buffer->get< particles_t >( ii )->getCApiPtr(),
                copy_of_output_buffer.get< particles_t >( ii )->getCApiPtr() ) );

            ::NS(Particles_random_init)(
                ptr_output_buffer->get< particles_t >( ii )->getCApiPtr() );

            SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)(
                ptr_output_buffer->get< particles_t >( ii )->getCApiPtr(),
                copy_of_output_buffer.get< particles_t >( ii )->getCApiPtr() ) );
        }

        /* ----------------------------------------------------------------- */
        /* Collect the buffers separately */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        job.collectParticles();
        ASSERT_TRUE( particles ==
            job.ptrParticlesBuffer()->get< particles_t >( 0 ) );

        particles_t* ptr_copy_particles = copy_of_pb.get< particles_t >( 0 );
        SIXTRL_ASSERT( ptr_copy_particles != nullptr );

        double const x0 = particles->getXValue( 0 );
        double const copy_x0 = ptr_copy_particles->getXValue( 0 );
        ASSERT_TRUE( EPS > std::fabs( x0 - copy_x0 ) );

        ASSERT_TRUE( ::NS(Particles_compare_values)( particles->getCApiPtr(),
            copy_of_pb.get< particles_t >( 0 )->getCApiPtr() ) == 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        job.collectBeamElements();

        ASSERT_TRUE( e01_drift ==
            job.ptrBeamElementsBuffer()->get< drift_t >( 0 ) );

        ASSERT_TRUE( ::NS(Drift_compare_values)( e01_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 0 )->getCApiPtr() ) == 0 );

        ASSERT_TRUE( e02_cavity ==
            job.ptrBeamElementsBuffer()->get< cavity_t >( 1 ) );

        ASSERT_TRUE( ::NS(Cavity_compare_values)( e02_cavity->getCApiPtr(),
            copy_of_eb.get< cavity_t >( 1 )->getCApiPtr() ) == 0 );

        ASSERT_TRUE( e03_drift ==
            job.ptrBeamElementsBuffer()->get< drift_t >( 2 ) );

        ASSERT_TRUE( ::NS(Drift_compare_values)( e03_drift->getCApiPtr(),
            copy_of_eb.get< drift_t >( 2 )->getCApiPtr() ) == 0 );

        /* Check that the collected output has output addresses */

        ASSERT_TRUE( e06_monitor ==
            job.ptrBeamElementsBuffer()->get< be_monitor_t >( 5 ) );

        ASSERT_TRUE( e09_monitor ==
            job.ptrBeamElementsBuffer()->get< be_monitor_t >( 8 ) );

        ASSERT_TRUE( e06_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );
        ASSERT_TRUE( e09_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );

        ASSERT_TRUE( copy_of_eb.get< be_monitor_t >( 5 )->getOutAddress() ==
                     e06_monitor->getOutAddress() );

        ASSERT_TRUE( copy_of_eb.get< be_monitor_t >( 8 )->getOutAddress() ==
                     e09_monitor->getOutAddress() );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        job.collectOutput();

        for( size_t ii = 0u ; ii < ptr_output_buffer->getNumObjects() ; ++ii )
        {
            ASSERT_TRUE( 0 == ::NS(Particles_compare_values)(
                ptr_output_buffer->get< particles_t >( ii )->getCApiPtr(),
                copy_of_output_buffer.get< particles_t >( ii )->getCApiPtr() ) );
        }

        /* ----------------------------------------------------------------- */
        /* Alter some values -> they will be pushed to the device */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        std::fill( particles->getAtElementId(),
                   particles->getAtElementId() + NUM_PARTICLES,
                   particle_index_t{ 42 } );

        SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)(
            particles->getCApiPtr(),
            copy_of_pb.get< particles_t >( 0 )->getCApiPtr() ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        e02_cavity->setLag( double{ 2.0 } );
        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity->getCApiPtr(),
            copy_of_eb.get< cavity_t >( 1 ) ) != 0 );

        SIXTRL_ASSERT( 0 == ::NS(LimitRect_compare_values)(
            e04_limit_rect->getCApiPtr(),
            copy_of_eb.get< limit_rect_t >( 3 )->getCApiPtr() ) );

        e04_limit_rect->setMinY( double{ 0.0 } );
        e04_limit_rect->setMaxY( double{ 0.5 } );
        SIXTRL_ASSERT( 0 != ::NS(LimitRect_compare_values)(
            e04_limit_rect->getCApiPtr(),
            copy_of_eb.get< limit_rect_t >( 3 ) ) );

        e06_monitor->setOutAddress( ::NS(be_monitor_addr_t){  42u } );
        e09_monitor->setOutAddress( ::NS(be_monitor_addr_t){ 137u } );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        for( size_t ii = 0u ; ii < ptr_output_buffer->getNumObjects() ; ++ii )
        {
            auto ptr = ptr_output_buffer->get< particles_t >( ii )->getState();
            particle_index_t const state_val =
                -( static_cast< particle_index_t >( ii + 1 ) );

            std::fill( ptr, ptr + NUM_PARTICLES, state_val );
            SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)(
                ptr_output_buffer->get< particles_t >( ii )->getCApiPtr(),
                copy_of_output_buffer.get< particles_t >( ii )->getCApiPtr() ) );
        }

        /* ----------------------------------------------------------------- */
        /* Push buffers to the device */

        job.pushParticles();
        job.pushBeamElements();
        job.pushOutput();

        /* ----------------------------------------------------------------- */
        /* Reset the changes on the host side so we can verify that the push
         * worked by collecting them again */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        std::fill( particles->getAtElementId(),
                   particles->getAtElementId() + NUM_PARTICLES,
                   particle_index_t{ 0 } );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        e02_cavity->setLag( copy_of_eb.get< cavity_t >( 1 )->getLag() );

        e04_limit_rect->setMinY( copy_of_eb.get< limit_rect_t >( 3 )->getMinY() );
        e04_limit_rect->setMaxY( copy_of_eb.get< limit_rect_t >( 3 )->getMaxY() );

        e06_monitor->setOutAddress( 0u );
        e09_monitor->setOutAddress( 0u );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        for( size_t ii = 0u ; ii < ptr_output_buffer->getNumObjects() ; ++ii )
        {
            auto ptr = ptr_output_buffer->get< particles_t >( ii )->getState();
            particle_index_t const state_val = particle_index_t{ 0 };
            std::fill( ptr, ptr + NUM_PARTICLES, state_val );
        }

        /* ----------------------------------------------------------------- */
        /* Collect the buffers again -> this should overwrite the locally
         * modified values again with those that we set before pushing: */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        job.collectParticles();
        ASSERT_TRUE( particles ==
            job.ptrParticlesBuffer()->get< particles_t >( 0 ) );

        ASSERT_TRUE( std::all_of( particles->getAtElementId(),
            particles->getAtElementId() + NUM_PARTICLES,
            []( particle_index_t const elem_id ){
                return ( elem_id == particle_index_t{ 42 } ); } ) );

        std::fill( copy_of_pb.get< particles_t >( 0 )->getAtElementId(),
           copy_of_pb.get< particles_t >( 0 )->getAtElementId() + NUM_PARTICLES,
           particle_index_t{ 42 } );

        ASSERT_TRUE( 0 == ::NS(Particles_compare_values)(
            particles->getCApiPtr(),
            copy_of_pb.get< particles_t >( 0 )->getCApiPtr() ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        job.collectBeamElements();

        ASSERT_TRUE( e02_cavity ==
            job.ptrBeamElementsBuffer()->get< cavity_t >( 1 ) );

        ASSERT_TRUE( EPS > std::fabs( e02_cavity->getLag() - double{ 2.0 } ) );

        copy_of_eb.get< cavity_t >( 1 )->setLag( double{ 2.0 } );

        ASSERT_TRUE( 0 == ::NS(Cavity_compare_values)(
            e02_cavity->getCApiPtr(),
            copy_of_eb.get< cavity_t >( 1 )->getCApiPtr() ) );


        ASSERT_TRUE( e04_limit_rect ==
            job.ptrBeamElementsBuffer()->get< limit_rect_t >( 3 ) );

        ASSERT_TRUE( EPS > std::fabs(
            e04_limit_rect->getMinY() - double{ 0.0 } ) );

        ASSERT_TRUE( EPS > std::fabs(
            e04_limit_rect->getMaxY() - double{ 0.5 } ) );

        copy_of_eb.get< limit_rect_t >( 3 )->setMinY( double{ 0.0 } );
        copy_of_eb.get< limit_rect_t >( 3 )->setMaxY( double{ 0.5 } );

        ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)(
            e04_limit_rect->getCApiPtr(),
            copy_of_eb.get< limit_rect_t >( 3 )->getCApiPtr() ) );

        /* Check that the collected output has output addresses */

        ASSERT_TRUE( e06_monitor ==
            job.ptrBeamElementsBuffer()->get< be_monitor_t >( 5 ) );

        ASSERT_TRUE( e06_monitor->getOutAddress() ==
            ::NS(be_monitor_addr_t){ 42 } );

        copy_of_eb.get< be_monitor_t >( 5 )->setOutAddress(
            ::NS(be_monitor_addr_t){ 42 } );

        ASSERT_TRUE( 0 == ::NS(BeamMonitor_compare_values)(
            e06_monitor->getCApiPtr(),
            copy_of_eb.get< be_monitor_t >( 5 )->getCApiPtr() ) );

        ASSERT_TRUE( e09_monitor ==
            job.ptrBeamElementsBuffer()->get< be_monitor_t >( 8 ) );

        ASSERT_TRUE( e09_monitor->getOutAddress() ==
            ::NS(be_monitor_addr_t){ 137 } );

        copy_of_eb.get< be_monitor_t >( 8 )->setOutAddress(
            ::NS(be_monitor_addr_t){ 137 } );

        ASSERT_TRUE( 0 == ::NS(BeamMonitor_compare_values)(
            e09_monitor->getCApiPtr(),
            copy_of_eb.get< be_monitor_t >( 8 )->getCApiPtr() ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        job.collectOutput();

        for( size_t ii = 0u ; ii < ptr_output_buffer->getNumObjects() ; ++ii )
        {
            auto ptr = ptr_output_buffer->get< particles_t >( ii )->getState();
            particle_index_t const state_val =
                -( static_cast< particle_index_t >( ii + 1 ) );

            ASSERT_TRUE( std::all_of( ptr, ptr + NUM_PARTICLES,
                [state_val]( particle_index_t const state )
                { return state == state_val; } ) );
        }
    }
}

/* end: tests/sixtracklib/opencl/test_track_job_cl_collect_push_cxx.cpp */
