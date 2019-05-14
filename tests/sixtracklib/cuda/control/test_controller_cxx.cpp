#include "sixtracklib/cuda/controller.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( CXX_CudaControllerTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_ctrl_t = st::CudaController;
    using size_t      = cuda_ctrl_t::size_type;

    cuda_ctrl_t cuda_controller;

    if( cuda_controller.numAvailableNodes() > size_t{ 0 } )
    {
        ASSERT_TRUE( cuda_controller.hasDefaultNode() );

    }
    else
    {
        std::cout << "no cuda devices available -> skipping unit-test"
                  << std::endl;
    }
}

/* end: tests/sixtracklib/cuda/control/test_controller_cxx.cpp */
