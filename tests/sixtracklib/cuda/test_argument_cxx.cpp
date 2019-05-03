#include "sixtracklib/cuda/argument.hpp"

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
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/controller.hpp"

TEST( CXX_CudaArgumentTests, ArgumentCObjectBufferTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using argument_t   = st::CudaArgument;
    using controller_t = st::CudaController;
    using particles_t  = st::Particles;
    using buffer_t     = st::Buffer;
    using buf_size_t   = buffer_t::size_type;
    using ctrl_size_t  = controller_t::size_type;

    buf_size_t const NUM_PARTICLES = 100;
    buffer_t pb;

    particles_t* particles = pb.createNew< particles_t >( NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );

    controller_t  controller;
    argument_t particles_arg( &controller );

    ASSERT_TRUE( particles_arg.archId() == controller.archId() );
    ASSERT_TRUE( particles_arg.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( particles_arg.requiresArgumentBuffer() );

    ASSERT_TRUE( particles_arg.size() == ctrl_size_t{ 0 } );
    ASSERT_TRUE( particles_arg.capacity() == ctrl_size_t{ 0 } );
    ASSERT_TRUE( !particles_arg.hasArgumentBuffer() );

    ASSERT_TRUE( !particles_arg.usesCObjectsBuffer() );
    ASSERT_TRUE( !particles_arg.usesCObjectsCxxBuffer() );
    ASSERT_TRUE( !particles_arg.usesRawArgument() );

    bool success = particles_arg.send( pb );

    ASSERT_TRUE( success == st::CONTROLLER_STATUS_SUCCESS );
    ASSERT_TRUE( particles_arg.hasArgumentBuffer() );
    ASSERT_TRUE( particles_arg.hasCudaArgBuffer() );
    ASSERT_TRUE( particles_arg.cudaArgBuffer() != nullptr );
    ASSERT_TRUE( particles_arg.capacity() >= pb.size() );
    ASSERT_TRUE( particles_arg.size() == pb.size() );
    ASSERT_TRUE( particles_arg.usesCObjectsCxxBuffer() );
    ASSERT_TRUE( particles_arg.usesCObjectsBuffer() );
    ASSERT_TRUE( particles_arg.ptrCObjectsCxxBuffer() == &pb );
    ASSERT_TRUE( &particles_arg.cobjectsCxxBuffer() == &pb );
    ASSERT_TRUE( particles_arg.ptrCObjectsBuffer() == pb.getCApiPtr() );
    ASSERT_TRUE( !particles_arg.usesRawArgument() );
    ASSERT_TRUE(  particles_arg.ptrRawArgument() == nullptr );
    ASSERT_TRUE( particles_arg.cudaController() == &controller );
}

/* end: tests/sixtracklib/cuda/test_controller_cxx.cpp */
