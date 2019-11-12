#include "sixtracklib/cuda/controller.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"

TEST( CXX_CudaControllerTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_ctrl_t = st::CudaController;
    using size_t      = cuda_ctrl_t::size_type;

    cuda_ctrl_t cuda_controller;

    size_t const num_available_nodes = cuda_controller.numAvailableNodes();

    if( num_available_nodes > size_t{ 0 } )
    {
        ASSERT_TRUE( cuda_controller.hasDefaultNode() );

        ASSERT_TRUE( cuda_controller.hasSelectedNode() );
        ASSERT_TRUE( cuda_controller.selectedNodeIndex() !=
                     cuda_ctrl_t::UNDEFINED_INDEX );

        ASSERT_TRUE( cuda_controller.selectedNodeIndex() ==
                     cuda_controller.defaultNodeIndex() );

        ASSERT_TRUE( cuda_controller.hasRemapCObjectBufferKernel() );
        ASSERT_TRUE( cuda_controller.hasRemapCObjectBufferDebugKernel() );

        ASSERT_TRUE( !cuda_controller.canUnselectNode() );
        ASSERT_TRUE( cuda_controller.canChangeSelectedNode() );
        ASSERT_TRUE( cuda_controller.canDirectlyChangeSelectedNode() );

        ASSERT_TRUE( cuda_controller.readyForRunningKernel() );
        ASSERT_TRUE( cuda_controller.readyForSend() );
        ASSERT_TRUE( cuda_controller.readyForReceive() );
        ASSERT_TRUE( cuda_controller.readyForRemap() );
    }
    else
    {
        std::cout << "no cuda devices available -> skipping unit-test"
                  << std::endl;
    }
}


TEST( CXX_CudaControllerTests, CObjectsBufferRemapping )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_ctrl_t   = st::CudaController;
    using buffer_t      = st::Buffer;
    using generic_obj_t = ::NS(GenericObj);
    using type_id_t     = ::NS(object_type_id_t);
    using size_t        = cuda_ctrl_t::size_type;

    buffer_t obj_buffer;

    size_t constexpr num_d_values = size_t{ 10 };
    size_t constexpr num_e_values = size_t{ 10 };
    size_t constexpr num_obj = size_t{ 10 };

    for( size_t ii = size_t{ 0 } ; ii < num_obj ; ++ii )
    {
        type_id_t const type_id = static_cast< type_id_t >( ii );

        generic_obj_t const* ptr_obj = ::NS(GenericObj_new)(
            obj_buffer.getCApiPtr(), type_id, num_d_values, num_e_values );
        SIXTRL_ASSERT( ptr_obj != nullptr );
        ( void )ptr_obj;
    }

    SIXTRL_ASSERT( obj_buffer.getNumObjects() == num_obj );
    SIXTRL_ASSERT( obj_buffer.getSize() > size_t{ 0 } );

    size_t const slot_size = obj_buffer.getSlotSize();

    cuda_ctrl_t cuda_controller;

    if( cuda_controller.numAvailableNodes() > size_t{ 0 } )
    {
        ASSERT_TRUE( cuda_controller.hasSelectedNode() );
        ASSERT_TRUE( cuda_controller.hasDefaultNode() );

        void* cuda_managed_buffer = nullptr;

        ::cudaError_t err = ::cudaMalloc(
            &cuda_managed_buffer, obj_buffer.getSize() );

        SIXTRL_ASSERT( err == ::cudaSuccess );

        ASSERT_TRUE( st::ARCH_STATUS_SUCCESS ==
            cuda_controller.sendMemory( cuda_managed_buffer,
                obj_buffer.dataBegin< unsigned char const* >(),
                    obj_buffer.getSize() ) );

        ASSERT_TRUE( !cuda_controller.isRemapped(
            cuda_managed_buffer, slot_size ) );

        ASSERT_TRUE( st::ARCH_STATUS_SUCCESS ==
            cuda_controller.remap( cuda_managed_buffer, slot_size ) );

        ASSERT_TRUE( cuda_controller.isRemapped(
            cuda_managed_buffer, slot_size ) );

        err = ::cudaFree( cuda_managed_buffer );

        SIXTRL_ASSERT( err == ::cudaSuccess );
        cuda_managed_buffer = nullptr;
        ( void )err;
    }
    else
    {
        std::cout << "no cuda devices available -> skipping unit-test"
                  << std::endl;
    }
}

/* end: tests/sixtracklib/cuda/control/test_controller_cxx.cpp */
