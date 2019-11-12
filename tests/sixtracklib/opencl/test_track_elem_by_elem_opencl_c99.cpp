#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/track/track.h"

#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/argument.h"


namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool performElemByElemTrackTest(
            ::NS(ClContext)* SIXTRL_RESTRICT context,
            ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
            ::NS(Buffer)* SIXTRL_RESTRICT cmp_elem_by_elem_buffer,
            ::NS(buffer_size_t) const num_turns,
            double const abs_tolerance =
                std::numeric_limits< double >::epsilon() );
    }
}

TEST( C99_OpenCLTrackElemByElemTests, TrackElemByElemHostAndDeviceCompareDrifts)
{
    using real_t       = ::NS(particle_real_t);
    using part_index_t = ::NS(particle_index_t);
    using size_t       = ::NS(buffer_size_t);
    using num_elem_t   = ::NS(particle_num_elements_t);

    part_index_t const NUM_TURNS   = part_index_t{ 3 };
    size_t const NUM_BEAM_ELEMENTS = size_t{ 5 };
    real_t const ABS_TOLERANCE     = std::numeric_limits< real_t >::epsilon();

    /* --------------------------------------------------------------------- */
    /* Creating a minimal machine description */

    ::NS(Buffer)* eb = ::NS(Buffer_new)( size_t{ 0 } );

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        ::NS(Drift)* drift = ::NS(Drift_add)( eb, double{ 1.0 } );
        ASSERT_TRUE( drift != nullptr );
    }

    ::NS(Buffer)* in_particles_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_LHC_NO_BB_PARTICLES_DUMP) );

    ::NS(Buffer)* elem_by_elem_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    ::NS(Particles)* initial_state = ::NS(Particles_buffer_get_particles)(
        in_particles_buffer, size_t{ 0 } );

    ASSERT_TRUE( initial_state != nullptr );
    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)(
        initial_state ) > num_elem_t{ 0 } );

    ::NS(Particles)* particles = ::NS(Particles_add_copy)(
        elem_by_elem_buffer, initial_state );

    /* --------------------------------------------------------------------- */
    /* Create elem by elem config structure and init from data */

    ::NS(ElemByElemConfig) elem_by_elem_config;
    ::NS(ElemByElemConfig_preset)( &elem_by_elem_config );

    ASSERT_TRUE( 0 == ::NS(ElemByElemConfig_init)( &elem_by_elem_config,
        particles, eb, part_index_t{ 0 }, NUM_TURNS ) );

    size_t elem_by_elem_index_offset = size_t{ 0 };

    ASSERT_TRUE( ::NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
        &elem_by_elem_config, elem_by_elem_buffer,
            &elem_by_elem_index_offset ) == 0 );

    ASSERT_TRUE( elem_by_elem_index_offset == size_t{ 1 } );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)(
        elem_by_elem_buffer ) == size_t{ 2 } );

    ::NS(Particles)* elem_by_elem_particles = nullptr;

    elem_by_elem_particles = ::NS(Particles_buffer_get_particles)(
            elem_by_elem_buffer, elem_by_elem_index_offset );

    ASSERT_TRUE( elem_by_elem_particles != nullptr );

    ::NS(Particles)* final_state = ::NS(Particles_add_copy)(
        elem_by_elem_buffer, initial_state );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( elem_by_elem_buffer ) ==
                 size_t{ 3 } );

    particles =
        ::NS(Particles_buffer_get_particles)( elem_by_elem_buffer, 0 );

    elem_by_elem_particles =
        ::NS(Particles_buffer_get_particles)( elem_by_elem_buffer, 1 );

    final_state =
        ::NS(Particles_buffer_get_particles)( elem_by_elem_buffer, 2 );

    ASSERT_TRUE( ::NS(ElemByElemConfig_assign_output_buffer)(
        &elem_by_elem_config, elem_by_elem_buffer, 1u ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    /* --------------------------------------------------------------------- */
    /* Track element by element on the host: */

    ASSERT_TRUE( 0 == ::NS(Track_all_particles_element_by_element_until_turn)(
        particles, &elem_by_elem_config, eb, NUM_TURNS ) );

    ASSERT_TRUE( ::NS(Particles_copy)( final_state, particles ) ==
                 ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(Particles_copy)( particles, initial_state ) ==
                 ::NS(ARCH_STATUS_SUCCESS) );

    ::NS(Buffer_delete)( in_particles_buffer );
    in_particles_buffer = nullptr;
    initial_state = nullptr;

    /* ===================================================================== */
    /* Init OpenCL context: */

    ::NS(ClContext)* ctx = ::NS(ClContext_create)();

    ASSERT_TRUE( ctx != nullptr );

    size_t const num_available_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( ctx );

    ::NS(ClContext_delete)( ctx );
    ctx = nullptr;

    std::vector< size_t > nodes;

    for( size_t node_index = size_t{ 0 } ;
            node_index < num_available_nodes ; ++node_index )
    {
        nodes.push_back( node_index );
    }

    if( !nodes.empty() )
    {
        bool const debug_mode[] = { false, true, false, true };
        bool const optimized[]  = { false, false, true, true };

        for( size_t const node_index : nodes )
        {
            for( size_t ii = size_t{ 0 } ; ii < size_t{ 4 } ; ++ii )
            {
                auto start_create = std::chrono::system_clock::now();
                ctx = ::NS(ClContext_create)();
                auto end_create = std::chrono::system_clock::now();

                auto create_duration = std::chrono::duration_cast<
                    std::chrono::seconds >( end_create - start_create );

                if( debug_mode[ ii ] )
                {
                    ::NS(ClContextBase_enable_debug_mode)( ctx );
                }
                else
                {
                    ::NS(ClContextBase_disable_debug_mode)( ctx );
                }

                if( optimized[ ii ] )
                {
                    if( !::NS(ClContextBase_is_available_node_amd_platform)(
                            ctx, node_index ) )
                    {
                        ::NS(ClContext_enable_optimized_tracking)( ctx );
                        ASSERT_TRUE( ::NS(ClContext_uses_optimized_tracking)(
                            ctx ) );
                    }
                    else
                    {
                        /* WARNING: Workaround */
                        std::cout << "WORKAROUND: Skipping optimized tracking "
                                  << " for AMD platforms\r\n";

                        ::NS(ClContext_disable_optimized_tracking)( ctx );
                        ASSERT_TRUE( !::NS(ClContext_uses_optimized_tracking)(
                            ctx ) );
                    }
                }
                else
                {
                    ::NS(ClContext_disable_optimized_tracking)( ctx );
                }

                ASSERT_TRUE( debug_mode[ ii ] ==
                    ::NS(ClContextBase_is_debug_mode_enabled)( ctx ) );

                auto start_select = std::chrono::system_clock::now();
                bool success = ::NS(ClContextBase_select_node_by_index)(
                    ctx, node_index );
                auto end_select = std::chrono::system_clock::now();

                auto select_duration = std::chrono::duration_cast<
                    std::chrono::seconds >( end_select - start_select );

                ASSERT_TRUE( success );
                ASSERT_TRUE( ::NS(ClContextBase_has_selected_node)( ctx ) );

                ::NS(context_node_info_t) const* node_info =
                    ::NS(ClContextBase_get_selected_node_info)( ctx );

                ASSERT_TRUE( node_info != nullptr );
                ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( ctx ) );
                ASSERT_TRUE( ::NS(ClContext_has_track_elem_by_elem_kernel)( ctx ) );

                char id_str[ 32 ];

                ::NS(ClContextBase_get_selected_node_id_str)(
                    ctx, id_str, 32 );

                std::cout
                      << "# --------------------------------------------------"
                      << "----------------------------------------------------"
                      << "\r\n"
                      << "# Run Test on :: \r\n"
                      << "# ID          :: " << id_str << "\r\n"
                      << "# NAME        :: "
                      << ::NS(ComputeNodeInfo_get_name)( node_info )
                      << "\r\n" << "# PLATFORM    :: "
                      << ::NS(ComputeNodeInfo_get_platform)( node_info )
                      << "\r\n" << "# Debug mode  :: " << std::boolalpha
                      << ::NS(ClContextBase_is_debug_mode_enabled)( ctx )
                      << "\r\n" << "# Optimized   :: "
                      << ::NS(ClContext_uses_optimized_tracking)(
                        ctx ) << "\r\n" << std::noboolalpha << "# "
                      << std::endl;

                std::cout << "NS(ClContext_create)() : "
                          << create_duration.count() << " sec\r\n";

                std::cout << "NS(ClContextBase_select_node_by_index)() : "
                          << select_duration.count() << " sec\r\n";

                auto run_start = std::chrono::system_clock::now();
                success = sixtrack::tests::performElemByElemTrackTest(
                    ctx, eb, elem_by_elem_buffer, NUM_TURNS, ABS_TOLERANCE );
                auto run_end   = std::chrono::system_clock::now();

                auto run_duration = std::chrono::duration_cast<
                    std::chrono::seconds >( run_end - run_start );
                ASSERT_TRUE( success );

                std::cout << "performElemByElemTrackTest : "
                          << run_duration.count() << " sec"
                          << std::endl;

                ::NS(ClContext_delete)( ctx );
                ctx = nullptr;
            }
        }
    }
    else
    {
        std::cout << "Unable to run unit-test -> no OpenCL Nodes found \r\n"
                  << "--> skipping!" << std::endl;
    }

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( elem_by_elem_buffer );
}

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        bool performElemByElemTrackTest(
            ::NS(ClContext)* SIXTRL_RESTRICT context,
            ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
            ::NS(Buffer)* SIXTRL_RESTRICT cmp_elem_by_elem_buffer,
            ::NS(buffer_size_t) const num_turns,
            double const abs_tolerance )
        {
            namespace st = SIXTRL_CXX_NAMESPACE;
            using size_t = ::NS(buffer_size_t);

            bool success = false;

            ::NS(arch_status_t) status = ::NS(ARCH_STATUS_SUCCESS);
            size_t const num_elem_by_elem_objects =
                ::NS(Buffer_get_num_of_objects)( cmp_elem_by_elem_buffer );

            ::NS(Buffer)* pb = ::NS(Buffer_new)( 0u );
            ::NS(Buffer)* output_buffer = ::NS(Buffer_new)( 0u );

            if( ( num_elem_by_elem_objects >= size_t{ 3 } ) &&
                ( pb != nullptr ) && ( beam_elements_buffer != nullptr ) &&
                ( cmp_elem_by_elem_buffer != nullptr ) &&
                ( num_turns > size_t{ 0 } ) && ( context != nullptr ) &&
                ( ::NS(ClContextBase_has_selected_node)( context ) ) &&
                ( abs_tolerance >= double{ 0. } ) )
            {
                success = true;
            }

            ::NS(Particles)* particles                        = nullptr;
            ::NS(Particles) const* final_state                = nullptr;
            ::NS(Particles) const* cmp_elem_by_elem_particles = nullptr;

            if( success )
            {
                particles = ::NS(Particles_add_copy)( pb,
                    ::NS(Particles_buffer_get_const_particles)(
                        cmp_elem_by_elem_buffer, 0u ) );

                cmp_elem_by_elem_particles =
                    ::NS(Particles_buffer_get_const_particles)(
                        cmp_elem_by_elem_buffer, 1u );

                final_state = ::NS(Particles_buffer_get_const_particles)(
                    cmp_elem_by_elem_buffer, 2u );
            }

            size_t const ebe_size = ::NS(Particles_get_num_of_particles)(
                cmp_elem_by_elem_particles );

            success = ( ( final_state   != nullptr ) &&
                        ( particles     != nullptr ) &&
                        ( cmp_elem_by_elem_particles != nullptr ) &&
                        ( ebe_size > size_t{ 0 } ) );

            ::NS(Particles)* elem_by_elem_particles = nullptr;

            if( !success )
            {
                std::cout << "success 01 : " << ( int )success << std::endl;
            }

            size_t elem_by_elem_output_offset = size_t{ 0 };
            size_t beam_monitor_output_offset = size_t{ 0 };
            ::NS(particle_index_t) min_turn_id = -1;

            if( success )
            {
                status = ::NS(OutputBuffer_prepare)(
                    beam_elements_buffer, output_buffer,
                    particles, num_turns, &elem_by_elem_output_offset,
                    &beam_monitor_output_offset, &min_turn_id );


                success = ( status == ::NS(ARCH_STATUS_SUCCESS) );
            }

            if( !success )
            {
                std::cout << "success 02" << std::endl;
            }

            /* ------------------------------------------------------------- */
            /* Create ClArguments for beam elements & the particles buffer   */

            ::NS(ClArgument)* particles_buffer_arg = nullptr;
            ::NS(ClArgument)* beam_elements_arg    = nullptr;
            ::NS(ClArgument)* output_buffer_arg    = nullptr;

            if( success )
            {
                particles_buffer_arg =
                    ::NS(ClArgument_new_from_buffer)( pb, context );

                beam_elements_arg =
                    ::NS(ClArgument_new_from_buffer)(
                        beam_elements_buffer, context );

                output_buffer_arg = ::NS(ClArgument_new_from_buffer)(
                        output_buffer, context );

                success = ( ( particles_buffer_arg != nullptr ) &&
                            ( beam_elements_arg    != nullptr ) &&
                            ( output_buffer_arg    != nullptr ) );

                if( !success )
                {
                    std::cout << "status 03" << std::endl;
                }
            }

            /* ------------------------------------------------------------- */
            /* Track for num-turns without assigned beam-monitors -> should
             * not change the correctness of tracking at all */

            cl_mem elem_by_elem_config_arg =
                ::NS(ClContext_create_elem_by_elem_config_arg)( context );

            size_t const pset_indices[] = { size_t{ 0 } };

            ::NS(ElemByElemConfig) elem_by_elem_config;

            if( success )
            {
                status = ::NS(ClContext_init_elem_by_elem_config_arg)(
                    context, elem_by_elem_config_arg, &elem_by_elem_config,
                    ::NS(ClArgument_get_ptr_cobj_buffer)( particles_buffer_arg ),
                    size_t{ 1 }, &pset_indices[ 0 ],
                    ::NS(ClArgument_get_ptr_cobj_buffer)( beam_elements_arg ),
                    num_turns, 0u );

                success = ( status == ::NS(ARCH_STATUS_SUCCESS) );

                if( !success )
                {
                    std::cout << "status 04 = " << status << std::endl;
                }
            }

            if( success )
            {
                status = ::NS(ClContext_assign_particles_arg)(
                    context, particles_buffer_arg );

                success = ( status == st::ARCH_STATUS_SUCCESS );

                if( !success )
                {
                    std::cout << "status 05a : " << status << std::endl;
                }

                status = ::NS(ClContext_assign_particle_set_arg)( context, 0u,
                    ::NS(Particles_get_num_of_particles)( particles ) );

                success &= ( status == st::ARCH_STATUS_SUCCESS );

                if( !success )
                {
                    std::cout << "status 05b : " << status << std::endl;
                }

                status = ::NS(ClContext_assign_beam_elements_arg)(
                    context, beam_elements_arg );

                if( !success )
                {
                    std::cout << "status 05c : " << status << std::endl;
                }

                success &= ( status == st::ARCH_STATUS_SUCCESS );

                if( !success )
                {
                    std::cout << "status 05d : " << status << std::endl;
                }

                status = ::NS(ClContext_assign_output_buffer_arg)(
                    context, output_buffer_arg );

                success &= ( status == st::ARCH_STATUS_SUCCESS );

                if( !success )
                {
                    std::cout << "status 05e : " << status << std::endl;
                }

                status = ::NS(ClContext_assign_elem_by_elem_config_arg)(
                    context, elem_by_elem_config_arg );

                success &= ( status == st::ARCH_STATUS_SUCCESS );

                if( !success )
                {
                    std::cout << "status 05f : " << status << std::endl;
                }
            }

            if( success )
            {
                elem_by_elem_particles = ::NS(Particles_buffer_get_particles)(
                    output_buffer, elem_by_elem_output_offset );

                success  = ( elem_by_elem_particles != nullptr );
                success &= ( ::NS(Particles_get_num_of_particles)(
                    elem_by_elem_particles ) == static_cast<
                        ::NS(particle_num_elements_t) >( ebe_size ) );

                if( !success )
                {
                    std::cout << "status 06" << std::endl;
                }
            }

            if( success )
            {
                status = ::NS(ClContext_assign_elem_by_elem_output)(
                    context, elem_by_elem_output_offset );

                success = ( status == ::NS(ARCH_STATUS_SUCCESS) );

                if( !success )
                {
                    std::cout << "status 07 = " << status << std::endl;
                }
            }

            if( success )
            {
                ::NS(track_status_t) const track_status =
                    ::NS(ClContext_track_elem_by_elem)( context, num_turns );

                success = ( track_status == ::NS(TRACK_SUCCESS) );

                if( !success )
                {
                    std::cout << "track_status 08 : " << track_status << std::endl;
                }
            }

            if( success )
            {
                success = ::NS(ClArgument_read)( particles_buffer_arg, pb );
                particles = ::NS(Particles_buffer_get_particles)( pb, 0u );
                success &= ( particles != nullptr );

                if( !success )
                {
                    std::cout << "status 09" << std::endl;
                }
            }

            if( success )
            {
                success = ::NS(ClArgument_read)(
                    output_buffer_arg, output_buffer );

                elem_by_elem_particles = ::NS(Particles_buffer_get_particles)(
                    output_buffer, elem_by_elem_output_offset );

                success &= ( elem_by_elem_particles != nullptr );

                if( !success )
                {
                    std::cout << "status 10" << std::endl;
                }
            }

            if( success )
            {
                int const cmp_result = ::NS(Particles_compare_values_with_treshold)(
                    cmp_elem_by_elem_particles, elem_by_elem_particles,
                    abs_tolerance );

                if( cmp_result != 0 )
                {
                    std::cout << "cmp_result 11 : " << cmp_result << std::endl;
                    success = false;
                }

                if( !success )
                {
                    ::NS(Buffer)* diff_buffer = ::NS(Buffer_new)( 0u );
                    ::NS(Particles)* diff = ::NS(Particles_new)( diff_buffer,
                        ::NS(Particles_get_num_of_particles)( particles ) );

                    ::NS(Particles_calculate_difference)(
                        particles, final_state, diff );

                    std::cout << std::endl << "cmp_elem_by_elem_buffer = " << std::endl;
                    ::NS(Particles_print_out)( cmp_elem_by_elem_particles );

                    std::cout << std::endl << "elem_by_elem_particles = " << std::endl;
                    ::NS(Particles_print_out)( elem_by_elem_particles );

                    std::cout << std::endl << "diff = " << std::endl;
                    ::NS(Particles_print_out)( diff );

                    ::NS(Buffer_delete)( diff_buffer );
                    diff_buffer = nullptr;
                }
            }

            ::NS(ClContext_delete_elem_by_elem_config_arg)(
                context, elem_by_elem_config_arg );

            ::NS(ClArgument_delete)( particles_buffer_arg );
            ::NS(ClArgument_delete)( beam_elements_arg );
            ::NS(ClArgument_delete)( output_buffer_arg );

            ::NS(Buffer_delete)( pb );
            ::NS(Buffer_delete)( output_buffer );

            return success;
        }
    }
}

/* end: */
