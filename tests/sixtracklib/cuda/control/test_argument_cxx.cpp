#include "sixtracklib/cuda/argument.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/controller.hpp"
#include "sixtracklib/common/buffer.hpp"

TEST( CXX_CudaArgumentTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_addr_t = st::CudaAddress;
    using size_t      = cuda_addr_t::size_type;
    using status_t    = cuda_addr_t::status_t;
    using cuda_ctrl_t = cuda_addr_t::cuda_controller_t;
}

/* end: tests/sixtracklib/cuda/control/test_argument_cxx.cpp */
