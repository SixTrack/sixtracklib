#include "sixtracklib/common/control/node_controller_base.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_controller_base.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        class TestNodeControllerBase :
            public SIXTRL_CXX_NAMESPACE::NodeControllerBase
        {
            public:

            TestNodeControllerBase() :
                NodeControllerBase( SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CPU,
                    "cpu", "" )
            {

            }

            TestNodeControllerBase( TestNodeControllerBase const& ) = default;
            TestNodeControllerBase( TestNodeControllerBase&& ) = default;

            TestNodeControllerBase& operator=(
                TestNodeControllerBase const& ) = default;

            TestNodeControllerBase& operator=(
                TestNodeControllerBase&& ) = default;

            ~TestNodeControllerBase() = default;
        };
    }
}

TEST( C99_CommonNodeControllerBaseTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using test_ctrl_t = st::tests::TestNodeControllerBase;
//     using size_t = test_ctrl_t::size_type;
    using node_index_t = test_ctrl_t::node_index_t;

    test_ctrl_t* node_controller = new test_ctrl_t;

    ASSERT_TRUE( ::NS(Controller_get_num_available_nodes)(
        node_controller ) == node_index_t{ 0 } );

    ::NS(Controller_delete)( node_controller );
    node_controller = nullptr;
}

/* end: tests/sixtracklib/common/control/test_node_controller_base_cxx.cpp */
