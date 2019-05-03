#include "sixtracklib/cuda/controller.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( C99_CudaControllerTests, MinimalUsageTest )
{
    using controller_t = ::NS(CudaController);

    controller_t* controller = ::NS(CudaController_create)();

    ASSERT_TRUE( controller != nullptr );
    ASSERT_TRUE( ::NS(Controller_get_arch_id)( controller ) ==
                 ::NS(ARCHITECTURE_CUDA) );

    ASSERT_TRUE( ::NS(Controller_is_ready_to_receive)( controller ) );
    ASSERT_TRUE( ::NS(Controller_is_ready_to_send)( controller ) );
    ASSERT_TRUE( ::NS(Controller_is_ready_to_remap)( controller ) );

    ::NS(CudaController_delete)( controller );
    controller = nullptr;
}

/* end: tests/sixtracklib/cuda/test_controller_c99.cpp */
