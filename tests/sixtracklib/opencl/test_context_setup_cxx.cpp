#include "sixtracklib/opencl/context.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( CXX_OpenCL_ContextSetupTests, ClContextEnvVariablesSetupTests )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using context_t = st::ClContext;
    using size_t = context_t::size_type;

    size_t const NUM_OPENCL_NODES = context_t::NUM_ALL_NODES();

    if( NUM_OPENCL_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes available -> skip test case" << std::endl;
        return;
    }

    std::cout << "total num of OpenCL nodes       : "
              << NUM_OPENCL_NODES << "\r\n";
    std::cout << "printing all OpenCL nodes       : \r\n\r\n";


    context_t::PRINT_ALL_NODES();

    char const* env_variable_begin = std::getenv( "SIXTRACKLIB_DEVICES" );

    std::cout << "content of SIXTRACKLIB_DEVICES  : ";

    if( env_variable_begin != nullptr )
    {
        std::cout << env_variable_begin;
    }

    size_t const NUM_AVAILABLE_NODES = context_t::NUM_AVAILABLE_NODES(
        nullptr, "SIXTRACKLIB_DEVICES" );

    ASSERT_TRUE( NUM_AVAILABLE_NODES <= NUM_OPENCL_NODES );

    std::cout << "\r\n" << "num of available OpenCL nodes   : "
              << NUM_AVAILABLE_NODES << "\r\n";

    if( NUM_AVAILABLE_NODES > size_t{ 0 } )
    {
        std::cout << "printing available OpenCl nodes :\r\n\r\n";
        context_t::PRINT_AVAILABLE_NODES( nullptr, "SIXTRACKLIB_DEVICES" );
    }

    context_t context;

    ASSERT_TRUE( context.numAvailableNodes() == NUM_AVAILABLE_NODES );
    ASSERT_TRUE( context.has_remapping_program() );
    ASSERT_TRUE( context.remapping_program_id() >= 0 );

    std::cout << "remapping buffer program_id     : "
              << context.remapping_program_id() << "\r\n"
              << "remapping buffer program options: "
              << context.programCompileOptions( context.remapping_program_id() )
              << "\r\n" << std::endl;
}

/* end: tests/sixtracklib/opencl/test_context_setup_cxx.cpp */
