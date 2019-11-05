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

    context_t::PRINT_AVAILABLE_NODES();

    context_t::size_type const num_available_nodes =
        context_t::NUM_AVAILABLE_NODES();

    if( num_available_nodes > context_t::size_type{ 0 } )
    {
        std::vector< context_t::node_id_t > available_nodes(
            num_available_nodes );

        ASSERT_TRUE( num_available_nodes == context_t::GET_AVAILABLE_NODES(
            available_nodes.data(), available_nodes.size() ) );
    }

    context_t context;

    ASSERT_TRUE( context.hasRemappingProgram() );
    ASSERT_TRUE( context.remappingProgramId() >= 0 );

    std::cout << context.programCompileOptions( context.remappingProgramId() )
              << std::endl;
}

/* end: tests/sixtracklib/opencl/test_context_setup_cxx.cpp */
