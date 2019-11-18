#include "sixtracklib/opencl/context.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( C99_OpenCL_ContextSetupTests, ClContextEnvVariablesSetupTests )
{
    using size_t = ::NS(ctrl_size_t);

    size_t const NUM_OPENCL_NODES = ::NS(OpenCL_get_num_all_nodes)();

    if( NUM_OPENCL_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes available -> skip test case" << std::endl;
        return;
    }

    std::cout << "total num of OpenCL nodes       : "
              << NUM_OPENCL_NODES << "\r\n";
    std::cout << "printing all OpenCL nodes       : \r\n\r\n";


    ::NS(OpenCL_print_all_nodes)();

    char const* env_variable_begin = std::getenv( "SIXTRACKLIB_DEVICES" );

    std::cout << "content of SIXTRACKLIB_DEVICES  : ";

    if( env_variable_begin != nullptr )
    {
        std::cout << env_variable_begin;
    }

    size_t const NUM_AVAILABLE_NODES = ::NS(OpenCL_num_available_nodes)(
        "SIXTRACKLIB_DEVICES" );

    ASSERT_TRUE( NUM_AVAILABLE_NODES <= NUM_OPENCL_NODES );

    std::cout << "\r\n" << "num of available OpenCL nodes   : "
              << NUM_AVAILABLE_NODES << "\r\n";

    if( NUM_AVAILABLE_NODES > size_t{ 0 } )
    {
        std::cout << "printing available OpenCl nodes :\r\n\r\n";
        ::NS(OpenCL_print_available_nodes_detailed)(
            nullptr, "SIXTRACKLIB_DEVICES" );
    }

    ::NS(ClContext)* context = ::NS(ClContext_create)();

    ASSERT_TRUE( ::NS(ClContextBase_get_num_available_nodes)( context ) ==
                 NUM_AVAILABLE_NODES );

    ASSERT_TRUE( ::NS(ClContextBase_has_remapping_program)( context ) );
    ASSERT_TRUE( ::NS(ClContextBase_remapping_program_id)( context ) >= 0 );

    std::cout << "remapping buffer program_id     : "
              << ::NS(ClContextBase_remapping_program_id)( context )
              << "\r\n" << "remapping buffer program options: "
              << ::NS(ClContextBase_get_program_compile_options)( context,
                 ::NS(ClContextBase_remapping_program_id)( context ) )
              << "\r\n" << std::endl;

    ::NS(ClContext_delete)( context );
    context = nullptr;
}

/* end: tests/sixtracklib/opencl/test_context_setup_c99.cpp */
