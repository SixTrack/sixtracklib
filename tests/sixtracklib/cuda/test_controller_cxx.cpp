#include "sixtracklib/cuda/controller.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( CXX_CudaControllerTests, MinimalUsageTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using controller_t = st::CudaController;

    controller_t controller;

    ASSERT_TRUE( controller.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( controller.readyForSend() );
    ASSERT_TRUE( controller.readyForReceive() );
    ASSERT_TRUE( controller.readyForRemap() );
}

/* end: tests/sixtracklib/cuda/test_controller_cxx.cpp */
