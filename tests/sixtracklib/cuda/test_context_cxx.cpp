#include "sixtracklib/cuda/context.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/context/definitions.h"

TEST( CXX_CudaContextTests, MinimalUsageTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using context_t = st::CudaContext;

    context_t context;

    ASSERT_TRUE( context.type() == st::CONTEXT_TYPE_CUDA );
    ASSERT_TRUE( context.readyForSend() );
    ASSERT_TRUE( context.readyForReceive() );
    ASSERT_TRUE( context.readyForRemap() );
}

/* end: tests/sixtracklib/cuda/test_context_cxx.cpp */
