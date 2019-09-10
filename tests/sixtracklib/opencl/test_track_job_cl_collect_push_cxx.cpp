#include "sixtracklib/opencl/track_job_cl.h"

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
    using aperture_rect_t    = st::LimitRect;
    using aperture_ellipse_t = st::LimitEllipse;
    using node_info_t        = cl_context_t::node_info_t;
    using node_id_t          = cl_context_t::node_id_t;
    using node_info_iter_t   = node_info_t const*;
    using status_t           = st::arch_status_t;
    using size_t             = buffer_t::size_type;
    using be_monitor_turn_t  = be_monitor_t::turn_t;
    using be_monitor_addr_t  = be_monitor_t::address_t;
    using particle_index_t   = particles_t::index_t;

    buffer_t eb;
    buffer_t pb;

    particles_t* particles = pb.createNew< particles_t >( 100u );
    SIXTRL_ASSERT( particles != nullptr );
    ::NS(Particles_init_particle_ids)( particles->getCApiPtr() );

    drift_t*  e01_drift = eb.add< drift_t  >( double{ 1.0 } );

    cavity_t* e02_cavity = eb.add< cavity_t >(
        double{ 100e3 }, double{ 400e6 }, double { 0.0 } );

    drift_t* e03_drift = eb.add< drift_t >( double{ 3.0 } );

    double const e04_min_x = double{ -1.0 };
    double const e04_max_x = double{ +1.0 };
    double const e04_min_y = double{ -2.0 };
    double const e04_max_y = double{  2.0 };

    aperture_rect_t* e04_limit_rect = eb.add< aperture_rect_t >(
        e04_min_x, e04_max_x, e04_min_y, e04_max_y );

    drift_t* e05_drift = eb.add< drift_t >( double{ 5.0 } );

    be_monitor_turn_t const e06_num_stores = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e06_start_turn = be_monitor_turn_t{ 0 };
    be_monitor_turn_t const e06_skip_turns = be_monitor_turn_t{ 0 };

    bool const e06_monitor_is_rolling = false;
    bool const e06_monitor_is_turn_ordered = false;
    be_monitor_addr_t const out_addr = be_monitor_addr_t{ 0 };

    be_monitor_t* e06_monitor = eb.add< be_monitor_t >(
        e06_num_stores, e06_start_turn, e06_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e06_monitor_is_rolling, e06_monitor_is_turn_ordered );

    drift_t* e07_drift = eb.add< drift_t >( double{ 7.0 } );

    double const e08_x_half_axis = double{ 0.5  };
    double const e08_y_half_axis = double{ 0.35 };

    aperture_ellipse_t* e08_limit_ell = eb.add< aperture_ellipse_t >(
        e08_x_half_axis, e08_y_half_axis );

    be_monitor_turn_t const e09_num_stores = be_monitor_turn_t{ 5  };
    be_monitor_turn_t const e09_start_turn = be_monitor_turn_t{ 10 };
    be_monitor_turn_t const e09_skip_turns = be_monitor_turn_t{ 5  };

    bool const e09_monitor_is_rolling = true;
    bool const e09_monitor_is_turn_ordered = false;

    be_monitor_t* e09_monitor = eb.add< be_monitor_t >(
        e09_num_stores, e09_start_turn, e09_skip_turns, out_addr,
        particle_index_t{ 0 }, particle_index_t{ 0 },
        e09_monitor_is_rolling, e09_monitor_is_turn_ordered );

    SIXTRL_ASSERT( eb.getNumObjects() == size_t{ 9 } );

    /* the pointers to the beam elements have been invalidated during creation
     * due to buffer reallocations */

    e01_drift      = eb.get< drift_t >( 0 );
    e02_cavity     = eb.get< cavity_t >( 1 );
    e03_drift      = eb.get< drift_t >( 2 );
    e04_limit_rect = eb.get< aperture_rect_t >( 3 );
    e05_drift      = eb.get< drift_t >( 4 );
    e06_monitor    = eb.get< be_monitor_t >( 5 );
    e07_drift      = eb.get< drift_t >( 6 );
    e08_limit_ell  = eb.get< aperture_ellipse_t >( 7 );
    e09_monitor    = eb.get< be_monitor_t >( 8 );

    SIXTRL_ASSERT( e01_drift      != nullptr );
    SIXTRL_ASSERT( e02_cavity     != nullptr );
    SIXTRL_ASSERT( e03_drift      != nullptr );
    SIXTRL_ASSERT( e04_limit_rect != nullptr );
    SIXTRL_ASSERT( e05_drift      != nullptr );
    SIXTRL_ASSERT( e06_monitor    != nullptr );
    SIXTRL_ASSERT( e07_drift      != nullptr );
    SIXTRL_ASSERT( e08_limit_ell  != nullptr );
    SIXTRL_ASSERT( e09_monitor    != nullptr );

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

        track_job_t  job( device_id_str, pb, eb );
        ASSERT_TRUE( job.hasOutputBuffer() );
        ASSERT_TRUE( job.hasBeamMonitorOutput() );
        ASSERT_TRUE( job.ptrParticlesBuffer() == &pb );
        ASSERT_TRUE( job.ptrBeamElementsBuffer() == &eb );
        ASSERT_TRUE( job.ptrOutputBuffer() != nullptr );

        buffer_t copy_of_eb( eb );
        SIXTRL_ASSERT( copy_of_eb.getNumObjects() == eb.getNumObjects() );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 0 )            != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 0 )            != e01_drift );
        SIXTRL_ASSERT( copy_of_eb.get< cavity_t >( 1 )           != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< cavity_t >( 1 )           != e02_cavity );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 2 )            != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 2 )            != e03_drift );
        SIXTRL_ASSERT( copy_of_eb.get< aperture_rect_t >( 3 )    != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< aperture_rect_t >( 3 )    !=
                       e04_limit_rect );

        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 4 )            != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 4 )            != e05_drift );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 )       != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 )       !=
                       e06_monitor );

        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 6 )            != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< drift_t >( 6 )            != e07_drift );
        SIXTRL_ASSERT( copy_of_eb.get< aperture_ellipse_t >( 7 ) != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< aperture_ellipse_t >( 7 ) !=
                       e08_limit_ell );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 )       != nullptr );
        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 )       !=
                       e09_monitor );

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

        buffer_t const* ptr_output_buffer = job.ptrOutputBuffer();
        ASSERT_TRUE( ptr_output_buffer != nullptr );

        buffer_t copy_of_output_buffer( *ptr_output_buffer );
        SIXTRL_ASSERT( copy_of_output_buffer.getNumObjects() ==
                       ptr_output_buffer->getNumObjects() );

        /* Collect the buffers separately */

        job.collectParticles();
        job.collectBeamElements();
        job.collectOutput();

        /* Check that the collected output has output addresses */

        SIXTRL_ASSERT( e06_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );
        SIXTRL_ASSERT( e09_monitor->getOutAddress() != be_monitor_addr_t{ 0 } );

        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 5 )->getOutAddress() !=
                       e06_monitor->getOutAddress() );

        SIXTRL_ASSERT( copy_of_eb.get< be_monitor_t >( 8 )->getOutAddress() !=
                       e09_monitor->getOutAddress() );



    }


}

/* end: */
