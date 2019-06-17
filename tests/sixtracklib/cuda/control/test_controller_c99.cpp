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

        ASSERT_TRUE( ::NS(Controller_has_selected_node)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_get_selected_node_index)( cuda_ctrl ) !=
                     cuda_ctrl_t::UNDEFINED_INDEX );

        ASSERT_TRUE( ::NS(Controller_get_selected_node_index)( cuda_ctrl ) ==
                     ::NS(Controller_get_default_node_index)( cuda_ctrl ) );

        ASSERT_TRUE( ::NS(Controller_has_remap_cobject_buffer_kernel)(
            cuda_ctrl ) );

        ASSERT_TRUE( ::NS(Controller_has_remap_cobject_buffer_debug_kernel)(
            cuda_ctrl ) );

        ASSERT_TRUE( !::NS(Controller_can_unselect_node)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_can_change_selected_node)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_can_directly_change_selected_node)(
            cuda_ctrl ) );

        ASSERT_TRUE( ::NS(Controller_is_ready_to_run_kernel)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_is_ready_to_remap)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_is_ready_to_send)( cuda_ctrl ) );
        ASSERT_TRUE( ::NS(Controller_is_ready_to_receive)( cuda_ctrl ) );
    }
    else
    {
        std::cout << "no cuda devices available -> skipping unit-test"
                  << std::endl;
    }

    ::NS(Controller_delete)( cuda_ctrl );
    cuda_ctrl = nullptr;
}


TEST( C99_CudaControllerTests, CObjectsBufferRemapping )
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

    cuda_ctrl_t* cuda_ctrl = ::NS(CudaController_create)();
    ASSERT_TRUE( cuda_ctrl != nullptr );

    if( NS(Controller_get_num_available_nodes)( cuda_ctrl ) > size_t{ 0 } )
    {
        ASSERT_TRUE( NS(Controller_has_selected_node)( cuda_ctrl ) );
        ASSERT_TRUE( NS(Controller_has_default_node)( cuda_ctrl ) );

        void* cuda_managed_buffer = nullptr;

        ::cudaError_t err = ::cudaMalloc(
            &cuda_managed_buffer, obj_buffer.getSize() );

        SIXTRL_ASSERT( err == ::cudaSuccess );

        ASSERT_TRUE( st::ARCH_STATUS_SUCCESS ==
            ::NS(CudaController_send_memory)( cuda_ctrl, cuda_managed_buffer,
                obj_buffer.dataBegin< unsigned char const* >(),
                    obj_buffer.getSize() ) );

        ASSERT_TRUE( !::NS(CudaController_is_managed_cobject_buffer_remapped)(
            cuda_ctrl, cuda_managed_buffer, slot_size ) );

        ASSERT_TRUE( st::ARCH_STATUS_SUCCESS ==
            ::NS(CudaController_remap_managed_cobject_buffer)( cuda_ctrl,
                cuda_managed_buffer, slot_size ) );

        ASSERT_TRUE( ::NS(CudaController_is_managed_cobject_buffer_remapped)(
            cuda_ctrl, cuda_managed_buffer, slot_size ) );

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

    ::NS(Controller_delete)( cuda_ctrl );
    cuda_ctrl = nullptr;
}

/* end: tests/sixtracklib/cuda/control/test_controller_c99.cpp */
