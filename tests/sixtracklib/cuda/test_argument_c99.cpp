#include "sixtracklib/cuda/argument.h"

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

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/cuda/context.h"

TEST( C99_CudaArgumentTests, ArgumentCObjectBufferTest )
{
    using argument_t  = ::NS(CudaArgument);
    using context_t   = ::NS(CudaContext);
    using particles_t = ::NS(Particles);
    using buffer_t    = ::NS(Buffer);
    using buf_size_t  = ::NS(buffer_size_t);
    using ctx_size_t  = ::NS(context_size_t);

    buf_size_t const NUM_PARTICLES = 100;
    buffer_t* pb = ::NS(Buffer_new)( 0u );

    particles_t* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );

    context_t*  context = ::NS(CudaContext_create)();
    ASSERT_TRUE( context != nullptr );

    argument_t* particles_arg = ::NS(CudaArgument_new)( context );
    ASSERT_TRUE( particles_arg != nullptr );

    bool success = ::NS(CudaArgument_send_buffer)( particles_arg, pb );
    ASSERT_TRUE( success );

    ::NS(CudaArgument_delete)( particles_arg );
    ::NS(CudaContext_delete)( context );
    ::NS(Buffer_delete)( pb );

    context = nullptr;
    particles_arg = nullptr;
    particles = nullptr;
    pb = nullptr;
}

/* end: tests/sixtracklib/cuda/test_context_cxx.cpp */
