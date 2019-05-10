#include "sixtracklib/cuda/controller.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

TEST( CXX_CudaControllerTests, MinimalUsageTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using controller_t = st::CudaController;
    using node_index_t = controller_t::node_index_t;
    using node_id_t    = controller_t::node_id_t;
    using node_info_t  = controller_t::node_info_t;

    controller_t controller;

    ASSERT_TRUE( controller.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( controller.hasArchStr() );
    ASSERT_TRUE( !controller.archStr().empty() );
    ASSERT_TRUE( controller.ptrArchStr() != nullptr );

    ASSERT_TRUE( 0 == std::strcmp( controller.ptrArchStr(),
                   SIXTRL_ARCHITECTURE_CUDA_STR ) );

    int temp_num_devices = int{ -1 };
    ::cudaError_t err = ::cudaGetDeviceCount( &temp_num_devices );

    if( ( err == ::cudaSuccess ) && ( temp_num_devices > 0 ) )
    {
        node_index_t const cmp_num_devices = static_cast< node_index_t >(
            temp_num_devices );

        ASSERT_TRUE( cmp_num_devices == controller.numAvailableNodes() );

        ASSERT_TRUE( !controller.hasSelectedNode() );
        ASSERT_TRUE( !controller.readyForSend() );
        ASSERT_TRUE( !controller.readyForReceive() );
        ASSERT_TRUE( !controller.readyForRemap() );

    }
    else
    {
        ASSERT_TRUE( controller.numAvailableNodes() == node_index_t{ 0 } );
    }
}

/* end: tests/sixtracklib/cuda/test_controller_cxx.cpp */
