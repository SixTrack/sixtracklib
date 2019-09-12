#include "sixtracklib/cuda/track_job.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track_job.h"

TEST( C99_CudaTrackJobCollectPushTests, TestCollectAndPush )
{
    using track_job_t        = ::NS(CudaTrackJob);
    using controller_t       = ::NS(CudaController);
    using buffer_t           = ::NS(Buffer);
    using particles_t        = ::NS(Particles);
    using drift_t            = ::NS(Drift);
    using cavity_t           = ::NS(Cavity);
    using be_monitor_t       = ::NS(BeamMonitor);
    using limit_rect_t       = ::NS(LimitRect);
    using limit_ellipse_t    = ::NS(LimitEllipse);
    using node_id_t          = ::NS(NodeId);
    using node_info_t        = ::NS(CudaNodeInfo);
    using size_t             = ::NS(buffer_size_t);
    using be_monitor_turn_t  = ::NS(be_monitor_turn_t);
    using be_monitor_addr_t  = ::NS(be_monitor_addr_t);
    using particle_index_t   = ::NS(particle_index_t);

    double const EPS = std::numeric_limits< double >::epsilon();

    /* Initialize pseudo random generator to init some parameters during the
       test */

    std::random_device rnd_device;
    std::mt19937_64 prng( rnd_device() );
    std::uniform_real_distribution< double > dbl_dist( 0.0, 1.0 );

    auto gen = [ &prng, &dbl_dist ]() { return dbl_dist( prng ); };

    /* Create particles buffers */

    buffer_t* init_eb = ::NS(Buffer_new)( size_t{ 0 } );
    buffer_t* init_pb = ::NS(Buffer_new)( size_t{ 0 } );

    size_t constexpr NUM_PARTICLES = size_t{ 100 };

    particles_t* particles = ::NS(Particles_new)( init_pb, NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );
    ::NS(Particles_random_init)( particles );

    drift_t* e01_drift = ::NS(Drift_add)( init_eb, double{ 1.0 } );

    cavity_t* e02_cavity = ::NS(Cavity_add)( init_eb,
         double{ 100e3 }, double{ 400e6 }, double { 0.0 } );

    drift_t* e03_drift = ::NS(Drift_add)( init_eb, double{ 3.0 } );

    double const e04_min_x = double{ -1.0 };
    double const e04_max_x = double{ +1.0 };
    double const e04_min_y = double{ -2.0 };
    double const e04_max_y = double{  2.0 };

    limit_rect_t* e04_limit_rect = ::NS(LimitRect_add)( init_eb,
        e04_min_x, e04_max_x, e04_min_y, e04_max_y );

    drift_t* e05_drift = ::NS(Drift_add)( init_eb, double{ 5.0 } );

    be_monitor_turn_t const e06_num_stores = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e06_start_turn = be_monitor_turn_t{ 0 };
    be_monitor_turn_t const e06_skip_turns = be_monitor_turn_t{ 0 };

    bool const e06_monitor_is_rolling = false;
    bool const e06_monitor_is_turn_ordered = false;
    be_monitor_addr_t const out_addr = be_monitor_addr_t{ 0 };

    be_monitor_t* e06_monitor = ::NS(BeamMonitor_add)( init_eb,
        e06_num_stores, e06_start_turn, e06_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e06_monitor_is_rolling, e06_monitor_is_turn_ordered );

    drift_t* e07_drift = ::NS(Drift_add)( init_eb, double{ 7.0 } );

    double const e08_x_half_axis = double{ 0.5  };
    double const e08_y_half_axis = double{ 0.35 };

    limit_ellipse_t* e08_limit_ell = ::NS(LimitEllipse_add)( init_eb,
        e08_x_half_axis, e08_y_half_axis );

    be_monitor_turn_t const e09_num_stores = be_monitor_turn_t{ 5  };
    be_monitor_turn_t const e09_start_turn = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e09_skip_turns = be_monitor_turn_t{ 5  };

    bool const e09_monitor_is_rolling = true;
    bool const e09_monitor_is_turn_ordered = false;

    be_monitor_t* e09_monitor = ::NS(BeamMonitor_add)( init_eb,
        e09_num_stores, e09_start_turn, e09_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e09_monitor_is_rolling, e09_monitor_is_turn_ordered );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( init_eb ) == size_t{ 9 } );

    /* -------------------------------------------------------------------- */
    /* Retrieve list of available nodes */

    size_t const num_avail_nodes =
        ::NS(CudaTrackJob_get_num_available_nodes)();

    if( num_avail_nodes == size_t{ 0 } )
    {
        std::cerr << "[          ] [ INFO ] \r\n"
                  << "[          ] [ INFO ] "
                  << "!!!!!!!! No cuda nodes found -> skipping test !!!!!!\r\n"
                  << "[          ] [ INFO ]" << std::endl;

        return;
    }

    std::vector< node_id_t > avail_node_ids( num_avail_nodes );

    size_t const num_nodes =
        ::NS(CudaTrackJob_get_available_node_ids_list)(
            avail_node_ids.size(), avail_node_ids.data() );

    ASSERT_TRUE( num_nodes == num_avail_nodes );

    for( auto const& node_id : avail_node_ids )
    {
        /* ----------------------------------------------------------------- */
        /* Create per-device copies of the particle and beam elements buffer */

        buffer_t* pb = ::NS(Buffer_new_from_copy)( init_pb );
        particles = ::NS(Particles_buffer_get_particles)( pb, 0 );
        SIXTRL_ASSERT( particles != nullptr );

        buffer_t* eb = ::NS(Buffer_new_from_copy)( init_eb );
        e01_drift  = ::NS(BeamElements_buffer_get_drift)( eb, 0 );
        e02_cavity = ::NS(BeamElements_buffer_get_cavity)( eb, 1 );
        e03_drift  = ::NS(BeamElements_buffer_get_drift)( eb, 2 );
        e04_limit_rect = ::NS(BeamElements_buffer_get_limit_rect)( eb, 3 );
        e05_drift  = ::NS(BeamElements_buffer_get_drift)( eb, 4 );
        e06_monitor = ::NS(BeamElements_buffer_get_beam_monitor)( eb, 5 );
        e07_drift  = ::NS(BeamElements_buffer_get_drift)( eb, 6 );
        e08_limit_ell = ::NS(BeamElements_buffer_get_limit_ellipse)( eb, 7 );
        e09_monitor = ::NS(BeamElements_buffer_get_beam_monitor)( eb, 8 );

        ASSERT_TRUE( e01_drift      != nullptr );
        ASSERT_TRUE( e02_cavity     != nullptr );
        ASSERT_TRUE( e03_drift      != nullptr );
        ASSERT_TRUE( e04_limit_rect != nullptr );
        ASSERT_TRUE( e05_drift      != nullptr );
        ASSERT_TRUE( e06_monitor    != nullptr );
        ASSERT_TRUE( e07_drift      != nullptr );
        ASSERT_TRUE( e08_limit_ell  != nullptr );
        ASSERT_TRUE( e09_monitor    != nullptr );

        /* Create a track job on the current node */
        std::string const node_id_str = node_id.toString();

        track_job_t* job = ::NS(CudaTrackJob_new)( node_id_str.c_str(), pb, eb );

        ASSERT_TRUE( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJobNew_requires_collecting)( job ) );
        ASSERT_TRUE( ::NS(TrackJobNew_has_output_buffer)( job ) );
        ASSERT_TRUE( ::NS(TrackJobNew_has_beam_monitor_output)( job ) );
        ASSERT_TRUE( ::NS(TrackJobNew_get_particles_buffer)( job ) == pb );
        ASSERT_TRUE( ::NS(TrackJobNew_get_beam_elements_buffer)( job ) == eb );
        ASSERT_TRUE( ::NS(TrackJobNew_get_output_buffer)( job ) != nullptr );

        ASSERT_TRUE( job != nullptr );

        controller_t* ptr_ctrl = ::NS(CudaTrackJob_get_ptr_controller)( job );

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


        buffer_t* copy_of_eb = ::NS(Buffer_new_from_copy)( eb );
        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( copy_of_eb ) == ::NS(Buffer_get_num_of_objects)( eb ) );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 0 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 0 ) != e01_drift );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) != e02_cavity );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 2 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 2 ) != e03_drift );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) != e04_limit_rect );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 4 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 4 ) != e05_drift );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ) != e06_monitor );

        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 6 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 6 ) != e07_drift );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_limit_ellipse)( copy_of_eb, 7 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_limit_ellipse)( copy_of_eb, 7 ) != e08_limit_ell );

        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ) != nullptr );
        SIXTRL_ASSERT( ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ) != e09_monitor );

        /* The beam monitors on the host side should have output adddr == 0: */
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_out_address)( e06_monitor ) != be_monitor_addr_t{ 0 } );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_out_address)( e09_monitor ) != be_monitor_addr_t{ 0 } );

        SIXTRL_ASSERT( ::NS(BeamMonitor_get_out_address)(
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ) ) ==
                       ::NS(BeamMonitor_get_out_address)( e06_monitor ) );

        SIXTRL_ASSERT( ::NS(BeamMonitor_get_out_address)(
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ) ) ==
                       ::NS(BeamMonitor_get_out_address)( e09_monitor ) );

        buffer_t* copy_of_pb = ::NS(Buffer_new_from_copy)( pb );
        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( copy_of_pb ) == ::NS(Buffer_get_num_of_objects)( pb ) );
        SIXTRL_ASSERT( ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) != nullptr );
        SIXTRL_ASSERT( ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) != particles );

        buffer_t* ptr_output_buffer = ::NS(TrackJobNew_get_output_buffer)( job );
        ASSERT_TRUE( ptr_output_buffer != nullptr );
        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) > size_t{ 0 } );
        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ==
                     ::NS(TrackJobNew_get_num_beam_monitors)( job ) );

        buffer_t* copy_of_output_buffer =
            ::NS(Buffer_new_from_copy)( ptr_output_buffer );

        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( copy_of_output_buffer ) ==
                       ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) );

        /* ----------------------------------------------------------------- */
        /* Set some parameters on the host side to different values. These will
         * be overwritten with the original values when we collect */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */
        SIXTRL_ASSERT( ::NS(Particles_compare_values)( particles,
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) ) == 0 );
        std::generate( particles->x, particles->x + NUM_PARTICLES, gen );
        SIXTRL_ASSERT( ::NS(Particles_compare_values)( particles,
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) ) != 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */
        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e01_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 0 ) ) == 0 );
        ::NS(Drift_set_length)( e01_drift,
            double{0.5} * ::NS(Drift_get_length)( e01_drift ) );

        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e01_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 0 ) ) != 0 );

        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity,
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) == 0 );
        ::NS(Cavity_set_voltage)( e02_cavity, double{2.0} * ::NS(Cavity_get_voltage)( e02_cavity ) );
        ::NS(Cavity_set_frequency)( e02_cavity, double{2.0} * ::NS(Cavity_get_frequency)( e02_cavity ) );
        ::NS(Cavity_set_lag)( e02_cavity, double{ 1.0 } );
        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity,
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) != 0 );

        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e03_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 2 ) ) == 0 );
        ::NS(Drift_set_length)( e03_drift, double{ 0.0 } );
        SIXTRL_ASSERT( ::NS(Drift_compare_values)( e03_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 2 ) ) != 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output: */
        for( size_t ii = 0u ; ii < ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ; ++ii )
        {
            SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ),
                ::NS(Particles_buffer_get_particles)( copy_of_output_buffer, ii ) ) );

            ::NS(Particles_random_init)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ) );

            SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ),
                ::NS(Particles_buffer_get_particles)( copy_of_output_buffer, ii ) ) );
        }

        /* ----------------------------------------------------------------- */
        /* Collect the buffers separately */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        ::NS(TrackJobNew_collect_particles)( job );
        ASSERT_TRUE( particles == ::NS(Particles_buffer_get_particles)(
            ::NS(TrackJobNew_get_particles_buffer)( job ), 0 ) );

        particles_t* ptr_copy_particles =
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 );
        SIXTRL_ASSERT( ptr_copy_particles != nullptr );

        double const x0 = particles->x[ 0 ];
        double const copy_x0 = ptr_copy_particles->x[ 0 ];
        ASSERT_TRUE( EPS > std::fabs( x0 - copy_x0 ) );

        ASSERT_TRUE( ::NS(Particles_compare_values)( particles,
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) ) == 0 );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        ::NS(TrackJobNew_collect_beam_elements)( job );

        ASSERT_TRUE( e01_drift == ::NS(BeamElements_buffer_get_drift)(
            ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 0 ) );

        ASSERT_TRUE( ::NS(Drift_compare_values)( e01_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 0 ) ) == 0 );

        ASSERT_TRUE( e02_cavity ==
            ::NS(BeamElements_buffer_get_cavity)( ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 1 ) );

        ASSERT_TRUE( ::NS(Cavity_compare_values)( e02_cavity,
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) == 0 );

        ASSERT_TRUE( e03_drift ==
            ::NS(BeamElements_buffer_get_drift)( ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 2 ) );

        ASSERT_TRUE( ::NS(Drift_compare_values)( e03_drift,
            ::NS(BeamElements_buffer_get_drift)( copy_of_eb, 2 ) ) == 0 );

        /* Check that the collected output has output addresses */

        ASSERT_TRUE( e06_monitor ==
            ::NS(BeamElements_buffer_get_beam_monitor)( ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 5 ) );

        ASSERT_TRUE( e09_monitor ==
            ::NS(BeamElements_buffer_get_beam_monitor)( ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 8 ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( e06_monitor ) != be_monitor_addr_t{ 0 } );
        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( e09_monitor ) != be_monitor_addr_t{ 0 } );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)(
                ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ) ) !=
            ::NS(BeamMonitor_get_out_address)( e06_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)(
                ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ) ) !=
            ::NS(BeamMonitor_get_out_address)( e09_monitor ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        ::NS(TrackJobNew_collect_output)( job );

        for( size_t ii = 0u ; ii < ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ; ++ii )
        {
            ASSERT_TRUE( 0 == ::NS(Particles_compare_values)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ),
                ::NS(Particles_buffer_get_particles)( copy_of_output_buffer, ii ) ) );
        }

        /* ----------------------------------------------------------------- */
        /* Alter some values -> they will be pushed to the device */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        std::fill( particles->at_element_id,
            particles->at_element_id + NUM_PARTICLES, particle_index_t{ 42 } );

        SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)( particles,
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        ::NS(Cavity_set_lag)( e02_cavity, double{ 2.0 } );
        SIXTRL_ASSERT( ::NS(Cavity_compare_values)( e02_cavity,
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) != 0 );

        SIXTRL_ASSERT( 0 == ::NS(LimitRect_compare_values)( e04_limit_rect,
            ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) ) );

        ::NS(LimitRect_set_min_y)( e04_limit_rect, double{ 0.0 } );
        ::NS(LimitRect_set_max_y)( e04_limit_rect, double{ 0.5 } );
        SIXTRL_ASSERT( 0 != ::NS(LimitRect_compare_values)( e04_limit_rect,
            ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) ) );

        ::NS(BeamMonitor_set_out_address)(
            e06_monitor, ::NS(be_monitor_addr_t){  42u } );

        ::NS(BeamMonitor_set_out_address)(
            e09_monitor, ::NS(be_monitor_addr_t){ 137u } );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        for( size_t ii = 0u ; ii < ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ; ++ii )
        {
            auto ptr = ::NS(Particles_get_state)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ) );
            particle_index_t const state_val =
                -( static_cast< particle_index_t >( ii + 1 ) );

            std::fill( ptr, ptr + NUM_PARTICLES, state_val );
            SIXTRL_ASSERT( 0 != ::NS(Particles_compare_values)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ),
                ::NS(Particles_buffer_get_particles)( copy_of_output_buffer, ii ) ) );
        }

        /* ----------------------------------------------------------------- */
        /* Push buffers to the device */

        ::NS(TrackJobNew_push_particles)( job );
        ::NS(TrackJobNew_push_beam_elements)( job );
        ::NS(TrackJobNew_push_output)( job );

        /* ----------------------------------------------------------------- */
        /* Reset the changes on the host side so we can verify that the push
         * worked by collecting them again */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        std::fill( particles->at_element_id,
                   particles->at_element_id + NUM_PARTICLES,
                   particle_index_t{ 0 } );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        ::NS(Cavity_set_lag)( e02_cavity, ::NS(Cavity_get_lag)(
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) );

        ::NS(LimitRect_set_min_y)( e04_limit_rect, ::NS(LimitRect_get_min_y)(
            ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) ) );

        ::NS(LimitRect_set_max_y)( e04_limit_rect, ::NS(LimitRect_get_max_y)(
            ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) ) );

        ::NS(BeamMonitor_set_out_address)( e06_monitor, 0u );
        ::NS(BeamMonitor_set_out_address)( e09_monitor, 0u );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        for( size_t ii = 0u ; ii < ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ; ++ii )
        {
            auto ptr = ::NS(Particles_get_state)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ) );
            particle_index_t const state_val = particle_index_t{ 0 };
            std::fill( ptr, ptr + NUM_PARTICLES, state_val );
        }

        /* ----------------------------------------------------------------- */
        /* Collect the buffers again -> this should overwrite the locally
         * modified values again with those that we set before pushing: */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* particles: */

        ::NS(TrackJobNew_collect_particles)( job );
        ASSERT_TRUE( particles == ::NS(Particles_buffer_get_particles)(
            ::NS(TrackJobNew_get_particles_buffer)( job ), 0 ) );

        ASSERT_TRUE( std::all_of( particles->at_element_id,
            particles->at_element_id + NUM_PARTICLES,
            []( particle_index_t const elem_id ){
                return ( elem_id == particle_index_t{ 42 } ); } ) );

        ptr_copy_particles =
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 );

        SIXTRL_ASSERT( ptr_copy_particles != nullptr );

        std::fill(
           ::NS(Particles_get_at_element_id)( ptr_copy_particles ),
           ::NS(Particles_get_at_element_id)( ptr_copy_particles ) + NUM_PARTICLES,
           particle_index_t{ 42 } );

        ASSERT_TRUE( 0 == ::NS(Particles_compare_values)(
            particles,
            ::NS(Particles_buffer_get_particles)( copy_of_pb, 0 ) ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* beam elements: */

        ::NS(TrackJobNew_collect_beam_elements)( job );

        ASSERT_TRUE( e02_cavity == ::NS(BeamElements_buffer_get_cavity)(
            ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 1 ) );

        ASSERT_TRUE( EPS > std::fabs(
            ::NS(Cavity_get_lag)( e02_cavity ) - double{ 2.0 } ) );

        ::NS(Cavity_set_lag)( ::NS(BeamElements_buffer_get_cavity)(
            copy_of_eb, 1 ), double{ 2.0 } );

        ASSERT_TRUE( 0 == ::NS(Cavity_compare_values)( e02_cavity,
            ::NS(BeamElements_buffer_get_cavity)( copy_of_eb, 1 ) ) );


        ASSERT_TRUE( e04_limit_rect == ::NS(BeamElements_buffer_get_limit_rect)(
            ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 3 ) );

        ASSERT_TRUE( EPS > std::fabs(
            ::NS(LimitRect_get_min_y)( e04_limit_rect ) - double{ 0.0 } ) );

        ASSERT_TRUE( EPS > std::fabs(
            ::NS(LimitRect_get_max_y)( e04_limit_rect ) - double{ 0.5 } ) );

        ::NS(LimitRect_set_min_y)( ::NS(BeamElements_buffer_get_limit_rect)(
            copy_of_eb, 3 ), double{ 0.0 } );

        ::NS(LimitRect_set_max_y)( ::NS(BeamElements_buffer_get_limit_rect)(
            copy_of_eb, 3 ), double{ 0.5 } );

        ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)( e04_limit_rect,
            ::NS(BeamElements_buffer_get_limit_rect)( copy_of_eb, 3 ) ) );

        /* Check that the collected output has output addresses */

        ASSERT_TRUE( e06_monitor == ::NS(BeamElements_buffer_get_beam_monitor)(
            ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 5 ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( e06_monitor ) ==
            ::NS(be_monitor_addr_t){ 42 } );

        ::NS(BeamMonitor_set_out_address)(
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ),
            ::NS(be_monitor_addr_t){ 42 } );

        ASSERT_TRUE( 0 == ::NS(BeamMonitor_compare_values)(
            e06_monitor,
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 5 ) ) );

        ASSERT_TRUE( e09_monitor == ::NS(BeamElements_buffer_get_beam_monitor)(
            ::NS(TrackJobNew_get_beam_elements_buffer)( job ), 8 ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( e09_monitor ) ==
            ::NS(be_monitor_addr_t){ 137 } );

        ::NS(BeamMonitor_set_out_address)(
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ),
            ::NS(be_monitor_addr_t){ 137 } );

        ASSERT_TRUE( 0 == ::NS(BeamMonitor_compare_values)(
            e09_monitor,
            ::NS(BeamElements_buffer_get_beam_monitor)( copy_of_eb, 8 ) ) );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /* output buffer: */

        ::NS(TrackJobNew_collect_output)( job );

        for( size_t ii = 0u ; ii < ::NS(Buffer_get_num_of_objects)( ptr_output_buffer ) ; ++ii )
        {
            auto ptr = ::NS(Particles_get_state)(
                ::NS(Particles_buffer_get_particles)( ptr_output_buffer, ii ) );

            particle_index_t const state_val =
                -( static_cast< particle_index_t >( ii + 1 ) );

            ASSERT_TRUE( std::all_of( ptr, ptr + NUM_PARTICLES,
                [state_val]( particle_index_t const state )
                { return state == state_val; } ) );
        }

        /* Local cleanup */

        ::NS(TrackJobNew_delete)( job );
        ::NS(Buffer_delete)( eb );
        ::NS(Buffer_delete)( copy_of_eb );
        ::NS(Buffer_delete)( pb );
        ::NS(Buffer_delete)( copy_of_pb );
        ::NS(Buffer_delete)( copy_of_output_buffer );
    }

    ::NS(Buffer_delete)( init_eb );
    ::NS(Buffer_delete)( init_pb );
}

/* end: tests/sixtracklib/cuda/track/test_track_job_collect_push_c99.cpp */
