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
#include "sixtracklib/common/context.h"

TEST( C99_CudaContextTests, MinimalUsageTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using context_t = ::NS(CudaContext);

    context_t* context = ::NS(CudaContext_create)();

    ASSERT_TRUE( context != nullptr );
    ASSERT_TRUE( ::NS(Context_get_type_id)( context ) ==
                 ::NS(CONTEXT_TYPE_CUDA) );

    ASSERT_TRUE( ::NS(Context_is_ready_to_receive)( context ) );
    ASSERT_TRUE( ::NS(Context_is_ready_to_send)( context ) );
    ASSERT_TRUE( ::NS(Context_is_ready_to_remap)( context ) );

    ::NS(CudaContext_delete)( context );
    context = nullptr;
}

/* end: tests/sixtracklib/cuda/test_context_cxx.cpp */
