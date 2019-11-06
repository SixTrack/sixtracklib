#include "sixtracklib/opencl/context.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( C99_OpenCL_ContextSetupTests, ClContextEnvVariablesSetupTests )
{
    using context_t = ::NS(ClContext);

    ::NS(OpenCL_print_available_nodes)();

    ::NS(arch_size_t) const num_available_nodes =
        ::NS(OpenCL_num_available_nodes)();

    if( num_available_nodes > ::NS(arch_size_t){ 0 } )
    {
        std::vector< ::NS(ComputeNodeId) > available_nodes(
            num_available_nodes );

        ASSERT_TRUE( num_available_nodes == ::NS(OpenCL_get_available_nodes)(
            available_nodes.data(), available_nodes.size() ) );
    }

    context_t* context = NS(ClContext_create)();

    ASSERT_TRUE( ::NS(ClContextBase_has_remapping_program)( context ) );
    ASSERT_TRUE( ::NS(ClContextBase_remapping_program_id)( context ) >= 0 );

    std::cout << ::NS(ClContextBase_get_program_compile_options)( context,
                    ::NS(ClContextBase_remapping_program_id)( context ) )
              << std::endl;

    ::NS(ClContextBase_delete)( context );
}

/* end: tests/sixtracklib/opencl/test_context_setup_c99.cpp */
