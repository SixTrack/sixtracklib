#include "sixtracklib/cuda/controller.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( C99_CudaControllerTests, BasicUsage )
{
    using cuda_ctrl_t = ::NS(CudaController);
    using size_t = ::NS(arch_size_t);

    cuda_ctrl_t* cuda_ctrl = ::NS(CudaController_create)();

    ASSERT_TRUE( cuda_ctrl != nullptr );

    if( ::NS(Controller_get_num_available_nodes)( cuda_ctrl ) > size_t{ 0 } )
    {
        ASSERT_TRUE( ::NS(Controller_has_default_node)( cuda_ctrl ) );

    }
    else
    {
        std::cout << "no cuda devices available -> skipping unit-test"
                  << std::endl;
    }
}

/* end: tests/sixtracklib/cuda/control/test_controller_c99.cpp */
