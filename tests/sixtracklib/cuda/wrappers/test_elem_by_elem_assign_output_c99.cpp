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
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/particles.h"

#include "sixtracklib/cuda/control/default_kernel_config.h"
#include "sixtracklib/cuda/control/kernel_config.h"
#include "sixtracklib/cuda/control/node_info.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/cuda/argument.h"

TEST( C99_CudaWrappersElemByElemAssignOutputTests,
      TestDefaultElemByElemConfig )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t           = ::NS(Particles);
    using c_buffer_t            = ::NS(Buffer);
    using elem_by_elem_config_t = ::NS(ElemByElemConfig);

    using cuda_ctrl_t           = ::NS(CudaController);
    using cuda_arg_t            = ::NS(CudaArgument);
    using cuda_kernel_conf_t    = ::NS(CudaKernelConfig);
    using cuda_node_info_t      = ::NS(CudaNodeInfo);

    using size_t                = ::NS(buffer_size_t);
    using pindex_t              = ::NS(particle_index_t);
    using status_t              = ::NS(arch_status_t);
    using node_index_t          = ::NS(node_index_t);
    using kernel_id_t           = ::NS(ctrl_kernel_id_t);

    c_buffer_t* particles_buffer     = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* output_buffer        = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* beam_elements_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    c_buffer_t* elem_by_elem_config_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    SIXTRL_ASSERT( beam_elements_buffer != nullptr );
    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( beam_elements_buffer ) >
        size_t{ 0 } );

    size_t const NUM_PARTICLES = size_t{ 10 };

    particles_t* particles = ::NS(Particles_new)(
        particles_buffer, NUM_PARTICLES );

    SIXTRL_ASSERT( particles != nullptr );

    /* Init particles  */
    pindex_t const min_at_turn_id = pindex_t{ 0 };

    ::NS(Particles_realistic_init)( particles );
    ::NS(Particles_set_all_at_turn_value)( particles, min_at_turn_id );

    size_t const UNTIL_TURN_ELEM_BY_ELEM = size_t{ 5 };

    elem_by_elem_config_t* elem_by_elem_conf =
        ::NS(ElemByElemConfig_new)( elem_by_elem_config_buffer );
    SIXTRL_ASSERT( elem_by_elem_conf != nullptr );

    /* -------------------------------------------------------------------- */
    /* Init the Cuda controller and arguments for the addresses
     * and the particles */

    cuda_ctrl_t* ctrl = ::NS(CudaController_create)();
    node_index_t const num_avail_nodes =
        ::NS(Controller_get_num_available_nodes)( ctrl );

    if( num_avail_nodes > node_index_t{ 0 } )
    {
        size_t num_processed_nodes = size_t{ 0 };

        std::vector< node_index_t > available_indices(
            num_avail_nodes, cuda_ctrl_t::UNDEFINED_INDEX );

        size_t const num_retrieved_node_indices =
            ::NS(Controller_get_available_node_indices)( ctrl,
                available_indices.size(), available_indices.data() );

        ASSERT_TRUE( num_retrieved_node_indices == available_indices.size() );

        for( node_index_t const ii : available_indices )
        {
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
            kernel_name +=
                "ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id_t const kernel_id =
                ::NS(CudaController_add_kernel_config_detailed)( ctrl,
                    kernel_name.c_str(), size_t{ 4 }, size_t{ 1 }, size_t{ 0 },
                        size_t{ 0 }, nullptr );

            cuda_kernel_conf_t* ptr_kernel_config =
                ::NS(CudaController_get_ptr_kernel_config)( ctrl, kernel_id );

            status_t status =
            ::NS(CudaKernelConfig_configure_assign_output_to_elem_by_elem_config_kernel)(
                ptr_kernel_config, ptr_node_info );

            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(ElemByElemConfig_preset)( elem_by_elem_conf );

            cuda_arg_t* elem_by_elem_conf_buffer_arg =
                ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( elem_by_elem_conf_buffer_arg != nullptr );

            cuda_arg_t* output_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( output_arg != nullptr );

            cuda_arg_t* result_arg = ::NS(CudaArgument_new)( ctrl );
            SIXTRL_ASSERT( result_arg != nullptr );

            size_t const num_particle_sets = size_t{ 1 };
            size_t const pset_indices_begin[] = { size_t{ 0 } };

            size_t output_buffer_index_offset =
                ::NS(Buffer_get_num_of_objects)( output_buffer ) + size_t{ 1 };

            status = ::NS(TestElemByElemConfigCtrlArg_prepare_assign_output_buffer)(
                particles_buffer, num_particle_sets, &pset_indices_begin[ 0 ],
                beam_elements_buffer, elem_by_elem_conf_buffer_arg,
                elem_by_elem_config_buffer, size_t{ 0 },
                output_arg, output_buffer, &output_buffer_index_offset,
                UNTIL_TURN_ELEM_BY_ELEM, result_arg );

            SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

            ::NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
                ptr_kernel_config, elem_by_elem_conf_buffer_arg, size_t{ 0 },
                    output_arg, output_buffer_index_offset, result_arg );

            status = ::NS(TestElemByElemConfigCtrlArg_evaluate_assign_output_buffer)(
                elem_by_elem_conf_buffer_arg, elem_by_elem_config_buffer,
                    size_t{ 0 }, output_arg, output_buffer,
                        output_buffer_index_offset, result_arg );

            ASSERT_TRUE( status == NS(ARCH_STATUS_SUCCESS) );

            /* ************************************************************* */

            ::NS(Argument_delete)( elem_by_elem_conf_buffer_arg );
            ::NS(Argument_delete)( output_arg );
            ::NS(Argument_delete)( result_arg );

            elem_by_elem_conf_buffer_arg = nullptr;
            result_arg = nullptr;
            output_arg = nullptr;

            ++num_processed_nodes;
        }

        ASSERT_TRUE( num_processed_nodes == static_cast< size_t >(
            num_avail_nodes ) );
    }
    else
    {
        std::cout << "No cuda nodes found -> skipping test" << std::endl;
    }

    ::NS(Buffer_delete)( particles_buffer );
    ::NS(Buffer_delete)( output_buffer );
    ::NS(Buffer_delete)( beam_elements_buffer );
    ::NS(Buffer_delete)( elem_by_elem_config_buffer );

    ::NS(Controller_delete)( ctrl );
}

/*end: tests/sixtracklib/cuda/wrappers/test_beam_monitor_assign_output_c99.cpp*/
