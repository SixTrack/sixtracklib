#include "sixtracklib/cuda/wrappers/track_job_wrappers.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

#include "sixtracklib/cuda/control/default_kernel_config.h"
#include "sixtracklib/cuda/control/kernel_config.h"
#include "sixtracklib/cuda/control/node_info.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/cuda/argument.h"

TEST( C99_CudaWrappersTrackParticlesUntilTurnTests,
      TrackSingleParticleSetAndCompareWithCpuTracking )
{
    using c_buffer_t         = ::NS(Buffer);
    using cuda_ctrl_t        = ::NS(CudaController);
    using cuda_arg_t         = ::NS(CudaArgument);
    using cuda_kernel_conf_t = ::NS(CudaKernelConfig);
    using cuda_node_info_t   = ::NS(CudaNodeInfo);
    using node_index_t       = ::NS(node_index_t);
    using kernel_id_t        = ::NS(ctrl_kernel_id_t);
    using particles_t        = ::NS(Particles);
    using buf_size_t         = ::NS(buffer_size_t);
    using track_status_t     = ::NS(track_status_t);
    using status_t           = ::NS(arch_status_t);

    double const ABS_TOLERANCE = double{ 1e-14 };

    c_buffer_t* in_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    particles_t* cmp_particles = ::NS(Particles_add_copy)(
        cmp_track_pb, ::NS(Particles_buffer_get_const_particles)(
            in_particles_buffer, buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN = buf_size_t{ 5 };

    track_status_t track_status = ::NS(TestTrackCpu_track_particles_until_cpu)(
        cmp_track_pb, buf_size_t{ 1 }, &track_pset_index, eb, UNTIL_TURN );

    SIXTRL_ASSERT( track_status == ::NS(TRACK_SUCCESS) );

    /* -------------------------------------------------------------------- */
    /* Init the Cuda controller and arguments for the addresses
     * and the particles */

    cuda_ctrl_t* ctrl = ::NS(CudaController_create)();
    node_index_t const num_avail_nodes =
        ::NS(Controller_get_num_available_nodes)( ctrl );

    if( num_avail_nodes > node_index_t{ 0 } )
    {
        buf_size_t num_processed_nodes = buf_size_t{ 0 };

        std::vector< node_index_t > available_indices(
            num_avail_nodes, cuda_ctrl_t::UNDEFINED_INDEX );

        buf_size_t const num_retrieved_node_indices =
            ::NS(Controller_get_available_node_indices)( ctrl,
                available_indices.size(), available_indices.data() );

        ASSERT_TRUE( num_retrieved_node_indices == available_indices.size() );

        for( node_index_t const ii : available_indices )
        {

            c_buffer_t* track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
            particles_t* particles = ::NS(Particles_add_copy)(
                track_pb, ::NS(Particles_buffer_get_const_particles)(
                    in_particles_buffer, buf_size_t{ 0 } ) );

            SIXTRL_ASSERT( particles != nullptr );

            if( ii != ::NS(Controller_get_selected_node_index)( ctrl ) )
            {
                ::NS(Controller_select_node_by_index)( ctrl, ii );
            }

            ASSERT_TRUE( ii == ::NS(Controller_get_selected_node_index)(
                ctrl ) );

            cuda_node_info_t const* ptr_node_info =
                ::NS(CudaController_get_ptr_node_info_by_index)( ctrl, ii );

            ASSERT_TRUE( ptr_node_info != nullptr );

            ::NS(NodeInfo_print_out)( ptr_node_info );

            /* ************************************************************* */

            std::string kernel_name = SIXTRL_C99_NAMESPACE_PREFIX_STR;
            kernel_name += "Track_particles_until_turn_cuda_wrapper";

            kernel_id_t const kernel_id =
                ::NS(CudaController_add_kernel_config_detailed)(
                    ctrl, kernel_name.c_str(), buf_size_t{ 5 },
                        buf_size_t{ 1 }, buf_size_t{ 0 }, buf_size_t{ 0 },
                            nullptr );

            cuda_kernel_conf_t* ptr_kernel_config =
                ::NS(CudaController_get_ptr_kernel_config)( ctrl, kernel_id );

            status_t status =
            ::NS(CudaKernelConfig_configure_track_until_turn_kernel)(
                ptr_kernel_config, ptr_node_info,
                    ::NS(Particles_get_num_of_particles)( particles ) );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            cuda_arg_t* particles_arg = ::NS(CudaArgument_new)( ctrl );
            cuda_arg_t* beam_elements_arg = ::NS(CudaArgument_new)( ctrl );
            cuda_arg_t* result_arg = ::NS(CudaArgument_new)( ctrl );

            status = ::NS(TestTrackCtrlArg_prepare_ctrl_arg_tracking)(
                particles_arg, track_pb, beam_elements_arg, eb, result_arg );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(Track_particles_until_turn_cuda_wrapper)(
                ptr_kernel_config, particles_arg, track_pset_index,
                    beam_elements_arg, UNTIL_TURN, result_arg );

            ASSERT_TRUE( ::NS(TestTrackCtrlArg_evaulate_ctrl_arg_tracking)(
                particles_arg, track_pb, buf_size_t{ 1 }, &track_pset_index,
                    cmp_track_pb, ABS_TOLERANCE, result_arg ) ==
                        ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(Argument_delete)( result_arg );
            ::NS(Argument_delete)( particles_arg );
            ::NS(Argument_delete)( beam_elements_arg );

            ::NS(Buffer_delete)( track_pb );

            ++num_processed_nodes;
        }

        ASSERT_TRUE( num_processed_nodes ==
            static_cast< size_t >( num_avail_nodes ) );
    }
    else
    {
        std::cout << "No cuda nodes found -> skipping test" << std::endl;
    }

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(Controller_delete)( ctrl );

    ::NS(Buffer_delete)( in_particles_buffer );
    ::NS(Buffer_delete)( cmp_track_pb );
    ::NS(Buffer_delete)( eb );
}

/*end:tests/sixtracklib/cuda/wrappers/test_track_particles_until_turn_c99.cpp*/
