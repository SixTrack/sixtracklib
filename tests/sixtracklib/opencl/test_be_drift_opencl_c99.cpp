#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"

#include "sixtracklib/opencl/internal/base_context.h"
#include "sixtracklib/opencl/argument.h"

namespace sixtrack
{
    namespace tests
    {
        bool performBeamElementCopyTest(
            ::st_ClContextBase* SIXTRL_RESTRICT context,
            int const program_id, int const kernel_id,
            ::st_Buffer* SIXTRL_RESTRICT in_buffer,
            double const abs_tolerance );
    }
}

TEST( C99_OpenCL_BeamElementsDriftTests, CopyDriftsHostToDeviceThenBackCompare )
{
    using size_t = ::st_buffer_size_t;

    ::st_Buffer* in_buffer   = ::st_Buffer_new( 0u );

    size_t const NUM_OBJECTS = size_t{ 1000 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        ::st_Drift* drift = ::st_Drift_add(
            in_buffer, static_cast< double >( ii ) );

        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( in_buffer  ) == NUM_OBJECTS );

    /* ---------------------------------------------------------------------- */
    /* Prepare path to test GPU kernel program file and compile options */

    std::ostringstream a2str;

    a2str <<  ::st_PATH_TO_BASE_DIR
          << "tests/sixtracklib/testlib/opencl/kernels/"
          << "opencl_beam_elements_opencl_kernel.cl";

    std::string const path_to_program = a2str.str();
    a2str.str( "" );

    a2str << " -D_GPUCODE=1"
          << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
          << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
          << " -I" << ::st_PATH_TO_SIXTRL_INCLUDE_DIR;

    if( std::strcmp( ::st_PATH_TO_SIXTRL_INCLUDE_DIR,
                     ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR ) != 0 )
    {
        a2str << " -I" << ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR;
    }

    std::string const program_compile_options( a2str.str() );
    a2str.str( "" );

    a2str << SIXTRL_C99_NAMESPACE_PREFIX_STR
          << "BeamElements_copy_beam_elements_opencl";

    std::string const kernel_name( a2str.str() );
    a2str.str( "" );

    /* ---------------------------------------------------------------------- */
    /* Get number of available devices */


    ::st_ClContextBase* context = ::st_ClContextBase_create();

    ASSERT_TRUE( context != nullptr );

    size_t const num_available_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    ::st_ClContextBase_delete( context );
    context = nullptr;

    for( size_t ii = size_t{ 0 } ; ii < num_available_nodes ; ++ii )
    {
        context = ::st_ClContextBase_create();
        ::st_ClContextBase_enable_debug_mode( context );

        ASSERT_TRUE(  ::st_ClContextBase_is_debug_mode_enabled( context ) );
        ASSERT_TRUE(  ::st_ClContextBase_select_node_by_index( context, ii ) );
        ASSERT_TRUE(  ::st_ClContextBase_has_selected_node( context ) );

        ::st_context_node_info_t const* node_info =
            ::st_ClContextBase_get_selected_node_info( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

        char id_str[ 32 ];
        ::st_ClContextBase_get_selected_node_id_str( context, id_str, 32 );

        int program_id = ::st_ClContextBase_add_program_file(
            context, path_to_program.c_str(), program_compile_options.c_str() );

        ASSERT_TRUE( program_id >= 0 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::st_ComputeNodeInfo_get_name( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::st_ComputeNodeInfo_get_platform( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( ::st_ClContextBase_is_program_compiled(
            context, program_id ) );

        int const kernel_id = ::st_ClContextBase_enable_kernel(
            context, kernel_name.c_str(), program_id );

        ASSERT_TRUE( kernel_id >= 0 );

        ASSERT_TRUE( sixtrack::tests::performBeamElementCopyTest(
            context, program_id, kernel_id, in_buffer,
                std::numeric_limits< double >::epsilon() ) );

        ::st_ClContextBase_delete( context );
        context = nullptr;
    }

    ::st_Buffer_delete( in_buffer );
}


TEST( C99_OpenCL_BeamElementsDriftTests, CopyDriftExactsHostToDeviceThenBackCompare )
{
    using size_t = ::st_buffer_size_t;

    ::st_Buffer* in_buffer   = ::st_Buffer_new( 0u );

    size_t const NUM_OBJECTS = size_t{ 1000 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        ::st_DriftExact* drift = ::st_DriftExact_add(
            in_buffer, static_cast< double >( ii ) );

        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( in_buffer  ) == NUM_OBJECTS );

    /* ---------------------------------------------------------------------- */
    /* Prepare path to test GPU kernel program file and compile options */

    std::ostringstream a2str;

    a2str <<  ::st_PATH_TO_BASE_DIR
          << "tests/sixtracklib/testlib/opencl/kernels/"
          << "opencl_beam_elements_opencl_kernel.cl";

    std::string const path_to_program = a2str.str();
    a2str.str( "" );

    a2str << " -D_GPUCODE=1"
          << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
          << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
          << " -I" << ::st_PATH_TO_SIXTRL_INCLUDE_DIR;

    if( std::strcmp( ::st_PATH_TO_SIXTRL_INCLUDE_DIR,
                     ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR ) != 0 )
    {
        a2str << " -I" << ::st_PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR;
    }

    std::string const program_compile_options( a2str.str() );
    a2str.str( "" );

    a2str << SIXTRL_C99_NAMESPACE_PREFIX_STR
          << "BeamElements_copy_beam_elements_opencl";

    std::string const kernel_name( a2str.str() );
    a2str.str( "" );

    /* ---------------------------------------------------------------------- */
    /* Get number of available devices */


    ::st_ClContextBase* context = ::st_ClContextBase_create();

    ASSERT_TRUE( context != nullptr );

    size_t const num_available_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    ::st_ClContextBase_delete( context );
    context = nullptr;

    for( size_t ii = size_t{ 0 } ; ii < num_available_nodes ; ++ii )
    {
        context = ::st_ClContextBase_create();
        ::st_ClContextBase_enable_debug_mode( context );

        ASSERT_TRUE(  ::st_ClContextBase_is_debug_mode_enabled( context ) );
        ASSERT_TRUE(  ::st_ClContextBase_select_node_by_index( context, ii ) );
        ASSERT_TRUE(  ::st_ClContextBase_has_selected_node( context ) );

        ::st_context_node_info_t const* node_info =
            ::st_ClContextBase_get_selected_node_info( context );

        ASSERT_TRUE( node_info != nullptr );
        ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

        char id_str[ 32 ];
        ::st_ClContextBase_get_selected_node_id_str( context, id_str, 32 );

        int program_id = ::st_ClContextBase_add_program_file(
            context, path_to_program.c_str(), program_compile_options.c_str() );

        ASSERT_TRUE( program_id >= 0 );

        std::cout << "# ------------------------------------------------------"
                  << "--------------------------------------------------------"
                  << "\r\n"
                  << "# Run Test on :: \r\n"
                  << "# ID          :: " << id_str << "\r\n"
                  << "# NAME        :: "
                  << ::st_ComputeNodeInfo_get_name( node_info ) << "\r\n"
                  << "# PLATFORM    :: "
                  << ::st_ComputeNodeInfo_get_platform( node_info ) << "\r\n"
                  << "# "
                  << std::endl;

        ASSERT_TRUE( ::st_ClContextBase_is_program_compiled(
            context, program_id ) );

        int const kernel_id = ::st_ClContextBase_enable_kernel(
            context, kernel_name.c_str(), program_id );

        ASSERT_TRUE( kernel_id >= 0 );

        ASSERT_TRUE( sixtrack::tests::performBeamElementCopyTest(
            context, program_id, kernel_id, in_buffer,
                std::numeric_limits< double >::epsilon() ) );

        ::st_ClContextBase_delete( context );
        context = nullptr;
    }

    ::st_Buffer_delete( in_buffer );
}

namespace sixtrack
{
    namespace tests
    {
        bool performBeamElementCopyTest(
            ::st_ClContextBase* SIXTRL_RESTRICT context,
            int const program_id, int const kernel_id,
            ::st_Buffer* SIXTRL_RESTRICT in_buffer,
            double const abs_tolerance )
        {
            bool success = false;

            if( ( in_buffer != nullptr ) && ( context != nullptr ) &&
                ( ::st_ClContextBase_has_selected_node( context ) ) &&
                ( ::st_ClContextBase_has_remapping_kernel( context ) ) &&
                ( program_id >= 0 ) &&
                ( static_cast< std::size_t >( program_id ) <
                    ::st_ClContextBase_get_num_available_programs( context ) ) &&
                ( kernel_id >= 0 ) &&
                ( static_cast< std::size_t >( kernel_id ) <
                     ::st_ClContextBase_get_num_available_kernels( context ) ) )
            {
                std::size_t const NUM_OBJECTS = ::st_Buffer_get_num_of_objects( in_buffer );

                ::st_Buffer* out_buffer = ::st_Buffer_new( 0u );

                success = ( 0 == ::st_BeamElements_copy_to_buffer( out_buffer,
                    ::st_Buffer_get_const_objects_begin( in_buffer ),
                    ::st_Buffer_get_const_objects_end( in_buffer ) ) );

                if( !success )
                {
                    ::st_Buffer_delete( out_buffer );
                    return success;
                }

                ::st_BeamElements_clear_buffer( out_buffer );

                ::st_ClArgument* in_buffer_arg =
                    ::st_ClArgument_new_from_buffer( in_buffer, context );

                success = ( ( in_buffer_arg != nullptr ) &&
                            ( ::st_ClArgument_get_argument_size( in_buffer_arg ) ==
                              ::st_Buffer_get_size( in_buffer ) ) &&
                            ( ::st_ClArgument_uses_cobj_buffer( in_buffer_arg ) ) &&
                            ( ::st_ClArgument_get_ptr_cobj_buffer( in_buffer_arg ) ==
                              in_buffer ) );

                if( !success )
                {
                    ::st_Buffer_delete( out_buffer );
                    return success;
                }

                ::st_ClArgument* out_buffer_arg =
                    ::st_ClArgument_new_from_buffer( out_buffer, context );

                success = ( ( out_buffer_arg != nullptr ) &&
                            ( ::st_ClArgument_get_argument_size( out_buffer_arg ) ==
                              ::st_Buffer_get_size( out_buffer ) ) &&
                            ( ::st_ClArgument_uses_cobj_buffer( out_buffer_arg ) ) &&
                            ( ::st_ClArgument_get_ptr_cobj_buffer( out_buffer_arg ) ==
                              out_buffer ) );

                if( !success )
                {
                    ::st_ClArgument_delete( in_buffer_arg );

                    if( out_buffer_arg != nullptr )
                    {
                        ::st_ClArgument_delete( out_buffer_arg );
                    }

                    ::st_Buffer_delete( out_buffer );

                    return success;
                }

                int32_t success_flag = int32_t{ 0 };

                ::st_ClArgument* success_flag_arg = ::st_ClArgument_new_from_memory(
                    &success_flag, sizeof( success_flag ), context );

                success = ( ( success_flag_arg != nullptr ) &&
                            ( ::st_ClArgument_get_argument_size( success_flag_arg ) ==
                                 sizeof( success_flag ) ) &&
                            ( !::st_ClArgument_uses_cobj_buffer( success_flag_arg ) ) );

                if( !success )
                {
                    ::st_ClArgument_delete( in_buffer_arg );
                    ::st_ClArgument_delete( out_buffer_arg );

                    if( success_flag_arg != nullptr )
                    {
                        ::st_ClArgument_delete( success_flag_arg );
                    }

                    ::st_Buffer_delete( out_buffer );
                }

                ::st_ClContextBase_assign_kernel_argument(
                    context, kernel_id, 0u, in_buffer_arg );

                ::st_ClContextBase_assign_kernel_argument(
                    context, kernel_id, 1u, out_buffer_arg );

                ::st_ClContextBase_assign_kernel_argument(
                    context, kernel_id, 2u, success_flag_arg );

                success = ::st_ClContextBase_run_kernel(
                    context, kernel_id, NUM_OBJECTS );

                if( success )
                {
                    success = ::st_ClArgument_read( out_buffer_arg, out_buffer );
                }

                if( success )
                {
                    success = ::st_ClArgument_read_memory( success_flag_arg,
                        &success_flag, sizeof( success_flag ) );
                }

                success &= ( success_flag == 0 );
                success &= ( ::st_Buffer_get_num_of_objects( out_buffer ) == NUM_OBJECTS );

                if( success )
                {
                    success = ( 0 == ::st_BeamElements_compare_lines_with_treshold(
                    ::st_Buffer_get_const_objects_begin( in_buffer ),
                    ::st_Buffer_get_const_objects_end( in_buffer ),
                    ::st_Buffer_get_const_objects_begin( out_buffer ), abs_tolerance ) );
                }

                ::st_ClArgument_delete( in_buffer_arg );
                ::st_ClArgument_delete( out_buffer_arg );
                ::st_ClArgument_delete( success_flag_arg );
                ::st_Buffer_delete( out_buffer );
            }

            return success;
        }
    }
}

/* end: tests/sixtracklib/opencl/test_be_drift_opencl_c99.cpp */
