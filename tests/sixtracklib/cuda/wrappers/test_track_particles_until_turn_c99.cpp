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
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
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
    ( void )cmp_particles;

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN = buf_size_t{ 5 };

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_until_turn_cpu)(
        cmp_track_pb, buf_size_t{ 1 }, &track_pset_index, eb, UNTIL_TURN );

    SIXTRL_ASSERT( track_status == ::NS(TRACK_SUCCESS) );
    ( void )track_status;

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
            SIXTRL_ASSERT( particles_arg != nullptr );

            cuda_arg_t* beam_elements_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( beam_elements_arg != nullptr );

            cuda_arg_t* result_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( result_arg != nullptr );

            status = ::NS(TestTrackCtrlArg_prepare_tracking)(
                particles_arg, track_pb, beam_elements_arg, eb, result_arg );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(Track_particles_until_turn_cuda_wrapper)(
                ptr_kernel_config, particles_arg, track_pset_index,
                    beam_elements_arg, UNTIL_TURN, result_arg );

            ASSERT_TRUE( ::NS(TestTrackCtrlArg_evaulate_tracking)(
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


TEST( C99_CudaWrappersTrackParticlesUntilTurnTests,
      TrackSingleParticleSetBeamMonitorsAndCompareWithCpuTracking )
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
    using pindex_t           = ::NS(particle_index_t);

    double const ABS_TOLERANCE = double{ 1e-14 };

    c_buffer_t* in_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    c_buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
    c_buffer_t* cmp_output_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    particles_t* cmp_particles = ::NS(Particles_add_copy)(
        cmp_track_pb, ::NS(Particles_buffer_get_const_particles)(
            in_particles_buffer, buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );
    ( void )cmp_particles;

    /* -------------------------------------------------------------------- */
    /* add end-of-turn beam-monitors */

    pindex_t const MIN_TURN_ID = pindex_t{ 0 };
    pindex_t const TURN_BY_TURN_START = MIN_TURN_ID;
    pindex_t const UNTIL_TURN_TURN_BY_TURN = pindex_t{ 10 } + MIN_TURN_ID;
    pindex_t const UNTIL_TURN = pindex_t{ 20 } + MIN_TURN_ID;
    pindex_t const SKIP_TURNS = ( UNTIL_TURN - MIN_TURN_ID ) / pindex_t{ 10 };

    status_t status = ::NS(BeamMonitor_insert_end_of_turn_monitors)(
        eb, TURN_BY_TURN_START, UNTIL_TURN_TURN_BY_TURN,
            UNTIL_TURN, SKIP_TURNS, ::NS(Buffer_get_objects_end)( eb ) );

    SIXTRL_ASSERT( status == NS(ARCH_STATUS_SUCCESS) );

    buf_size_t const num_beam_monitors = buf_size_t{ 2 };

    SIXTRL_ASSERT( ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) ==
                   num_beam_monitors );

    /* -------------------------------------------------------------------- */
    /* assign cmp_output_buffer to beam_monitors */

    buf_size_t const NUM_PSETS = buf_size_t{ 1 };
    buf_size_t track_pset_index = buf_size_t{ 0 };

    buf_size_t const UNTIL_TURN_ELEM_BY_ELEM = buf_size_t{ 0 };
    buf_size_t elem_by_elem_output_offset_index = buf_size_t{ 0 };
    buf_size_t beam_monitor_output_offset_index = buf_size_t{ 0 };
    pindex_t cmp_min_turn_id = pindex_t{ -1 };

    status = NS(OutputBuffer_prepare_for_particle_sets)(
        eb, cmp_output_buffer, cmp_track_pb, NUM_PSETS, &track_pset_index,
            UNTIL_TURN_ELEM_BY_ELEM, &elem_by_elem_output_offset_index,
                &beam_monitor_output_offset_index, &cmp_min_turn_id );

    SIXTRL_ASSERT( status == NS(ARCH_STATUS_SUCCESS) );
    SIXTRL_ASSERT( cmp_min_turn_id == MIN_TURN_ID );

    status = NS(BeamMonitor_assign_output_buffer_from_offset)(
        eb, cmp_output_buffer, MIN_TURN_ID, beam_monitor_output_offset_index );

    SIXTRL_ASSERT( status == NS(ARCH_STATUS_SUCCESS) );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_until_turn_cpu)(
        cmp_track_pb, NUM_PSETS, &track_pset_index, eb, UNTIL_TURN );

    SIXTRL_ASSERT( track_status == ::NS(TRACK_SUCCESS) );
    ( void )track_status;

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
            c_buffer_t* output_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

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
            kernel_name += "BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id_t const assign_kernel_id =
                ::NS(CudaController_add_kernel_config_detailed)( ctrl,
                    kernel_name.c_str(), buf_size_t{ 5 }, buf_size_t{ 1 },
                        buf_size_t{ 0 }, buf_size_t{ 0 }, nullptr );

            cuda_kernel_conf_t* ptr_assign_kernel_config =
                ::NS(CudaController_get_ptr_kernel_config)(
                    ctrl, assign_kernel_id );

            status =
            ::NS(CudaKernelConfig_configure_assign_output_to_beam_monitors_kernel)(
                ptr_assign_kernel_config, ptr_node_info, num_beam_monitors );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            kernel_name.clear();
            kernel_name = SIXTRL_C99_NAMESPACE_PREFIX_STR;
            kernel_name += "Track_particles_until_turn_cuda_wrapper";

            kernel_id_t const kernel_id =
                ::NS(CudaController_add_kernel_config_detailed)(
                    ctrl, kernel_name.c_str(), buf_size_t{ 5 },
                        buf_size_t{ 1 }, buf_size_t{ 0 }, buf_size_t{ 0 },
                            nullptr );

            cuda_kernel_conf_t* ptr_kernel_config =
                ::NS(CudaController_get_ptr_kernel_config)( ctrl, kernel_id );

            status = ::NS(CudaKernelConfig_configure_track_until_turn_kernel)(
                ptr_kernel_config, ptr_node_info,
                    ::NS(Particles_get_num_of_particles)( particles ) );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            cuda_arg_t* particles_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( particles_arg != nullptr );

            cuda_arg_t* beam_elements_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( beam_elements_arg != nullptr );

            cuda_arg_t* output_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( output_arg != nullptr );

            cuda_arg_t* result_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( result_arg != nullptr );

            buf_size_t output_buffer_index_offset =
                ::NS(Buffer_get_num_of_objects)( output_buffer ) + size_t{ 1 };

            /* reset the output addresses of the eb beam monitors to prepare a
             * clean slate for the gpu tracking */
            ::NS(BeamMonitor_reset_all_in_buffer)( eb );

            status = ::NS(TestTrackCtrlArg_prepare_tracking)(
                particles_arg, track_pb, beam_elements_arg, eb, result_arg );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* We accept some duplication of work between the preparation
             * for the tracking and the preparation for the beam monitor
             * output assignment: */

            status = ::NS(TestBeamMonitorCtrlArg_prepare_assign_output_buffer)(
                track_pb, NUM_PSETS, &track_pset_index, beam_elements_arg,
                eb, output_arg, output_buffer, MIN_TURN_ID,
                    &output_buffer_index_offset, result_arg );

            SIXTRL_ASSERT( status == NS(ARCH_STATUS_SUCCESS) );
            SIXTRL_ASSERT( output_buffer_index_offset ==
                           beam_monitor_output_offset_index );


            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
                ptr_assign_kernel_config, beam_elements_arg, output_arg,
                    MIN_TURN_ID, output_buffer_index_offset, result_arg );

            status = ::NS(TestBeamMonitorCtrlArg_evaluate_assign_output_buffer)(
                beam_elements_arg, eb, output_arg, output_buffer,
                output_buffer_index_offset, num_beam_monitors, result_arg );

            SIXTRL_ASSERT( status == NS(ARCH_STATUS_SUCCESS) );

            ::NS(Track_particles_until_turn_cuda_wrapper)(
                ptr_kernel_config, particles_arg, track_pset_index,
                    beam_elements_arg, UNTIL_TURN, result_arg );

            status = ::NS(TestTrackCtrlArg_evaulate_tracking)(
                particles_arg, track_pb, NUM_PSETS, &track_pset_index,
                    cmp_track_pb, ABS_TOLERANCE, result_arg );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            status = ::NS(TestTrackCtrlArg_evaluate_tracking_all)(
                output_arg, output_buffer, cmp_output_buffer, ABS_TOLERANCE,
                result_arg );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(Argument_delete)( result_arg );
            ::NS(Argument_delete)( output_arg );
            ::NS(Argument_delete)( beam_elements_arg );
            ::NS(Argument_delete)( particles_arg );

            result_arg = nullptr;
            output_arg = nullptr;
            beam_elements_arg = nullptr;
            particles_arg = nullptr;

            ::NS(Buffer_delete)( track_pb );
            ::NS(Buffer_delete)( output_buffer );

            track_pb = nullptr;
            output_buffer = nullptr;

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
    ::NS(Buffer_delete)( cmp_output_buffer );
    ::NS(Buffer_delete)( eb );
}

/*end:tests/sixtracklib/cuda/wrappers/test_track_particles_until_turn_c99.cpp*/
