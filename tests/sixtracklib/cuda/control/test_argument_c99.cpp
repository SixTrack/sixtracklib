#include "sixtracklib/cuda/argument.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"

TEST( CXX_CudaArgumentTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_arg_t    = ::NS(CudaArgument);
    using size_t        = ::NS(ctrl_size_t);
    using status_t      = ::NS(ctrl_status_t);
    using cuda_ctrl_t   = ::NS(CudaController);
    using c_buffer_t    = ::NS(Buffer);
    using generic_obj_t = ::NS(GenericObj);
    using type_id_t     = ::NS(object_type_id_t);

    /* --------------------------------------------------------------------- */
    /* Prepare the cobject buffer's (both C++ and C99) */

    size_t constexpr num_d_values = size_t{ 10 };
    size_t constexpr num_e_values = size_t{ 10 };
    size_t constexpr num_obj      = size_t{ 10 };

    c_buffer_t* c_obj_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    for( size_t ii = size_t{ 0 } ; ii < num_obj ; ++ii )
    {
        type_id_t const type_id = static_cast< type_id_t >( ii );

        generic_obj_t* ptr_c_obj = ::NS(GenericObj_new)(
            c_obj_buffer, type_id, num_d_values, num_e_values );

        SIXTRL_ASSERT( ptr_c_obj != nullptr );
    }

    size_t const slot_size = ::NS(Buffer_get_slot_size)( c_obj_buffer );
    SIXTRL_ASSERT( slot_size > size_t{ 0 } );

    c_buffer_t* c_cpy_buffer = ::NS(Buffer_new)(
        ::NS(Buffer_get_capacity)( c_obj_buffer ) );

    cuda_ctrl_t* cuda_controller = ::NS(CudaController_create)();

    cuda_arg_t* arg1 = ::NS(CudaArgument_new)( cuda_controller);
    ASSERT_TRUE( arg1 != nullptr );


    ASSERT_TRUE( !::NS(Argument_has_argument_buffer)( arg1 ) );
    ASSERT_TRUE( !::NS(CudaArgument_has_cuda_arg_buffer)( arg1 ) );
    ASSERT_TRUE(  ::NS(CudaArgument_get_cuda_arg_buffer)( arg1 ) == nullptr );

    ASSERT_TRUE( !::NS(Argument_uses_cobjects_buffer)( arg1 ) );
    ASSERT_TRUE(  ::NS(Argument_get_const_cobjects_buffer)( arg1 ) == nullptr );

    ASSERT_TRUE( !::NS(Argument_uses_raw_argument)( arg1 ) );
    ASSERT_TRUE(  ::NS(Argument_get_const_ptr_raw_argument)( arg1 ) == nullptr );

    status_t status = ::NS(Argument_send_buffer)( arg1, c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(Argument_has_argument_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(CudaArgument_has_cuda_arg_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(CudaArgument_get_cuda_arg_buffer)( arg1 ) != nullptr );

    ASSERT_TRUE( ::NS(Argument_uses_cobjects_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) ==
                 c_obj_buffer );

    ASSERT_TRUE(  ::NS(Argument_get_size)( arg1 ) > size_t{ 0 } );
    ASSERT_TRUE(  ::NS(Argument_get_capacity)( arg1 ) >=
                  ::NS(Argument_get_size)( arg1 ) );

    ASSERT_TRUE(  ::NS(Argument_get_size)( arg1 ) ==
                  ::NS(Buffer_get_size)( c_obj_buffer ) );

    ASSERT_TRUE(  ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        cuda_controller, ::NS(CudaArgument_get_cuda_arg_buffer)(
            arg1 ), slot_size ) );

    status = ::NS(Argument_receive_buffer)( arg1, c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( ::NS(Argument_has_argument_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(CudaArgument_has_cuda_arg_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(CudaArgument_get_cuda_arg_buffer)( arg1 ) != nullptr );

    ASSERT_TRUE( ::NS(Argument_uses_cobjects_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) ==
                 c_obj_buffer );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );

    ASSERT_TRUE(  ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        cuda_controller, ::NS(CudaArgument_get_cuda_arg_buffer)( arg1 ),
              slot_size ) );

    status = ::NS(Argument_receive_buffer)( arg1, c_cpy_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( ::NS(Argument_uses_cobjects_buffer)( arg1 ) );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg1 ) == c_obj_buffer );


    ASSERT_TRUE( ::NS(CudaController_is_managed_cobject_buffer_remapped)(
        cuda_controller, ::NS(CudaArgument_get_cuda_arg_buffer)( arg1 ),
            slot_size ) );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );
    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_size)( c_obj_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) );

    /* --------------------------------------------------------------------- */

    cuda_arg_t* arg2 = ::NS(CudaArgument_new_from_buffer)(
        c_obj_buffer, cuda_controller);

    ASSERT_TRUE( arg2 != nullptr );

    ASSERT_TRUE( ::NS(Argument_has_argument_buffer)( arg2 ) );
    ASSERT_TRUE( ::NS(CudaArgument_has_cuda_arg_buffer)( arg2 ) );
    ASSERT_TRUE( ::NS(Argument_uses_cobjects_buffer)( arg2 ) );

    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg2 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_cobjects_buffer)( arg2 ) == c_obj_buffer );
    ASSERT_TRUE( ::NS(Argument_get_size)( arg2 ) == ::NS(Buffer_get_size)( c_obj_buffer ) );
    ASSERT_TRUE( ::NS(Argument_get_capacity)( arg2 ) >= ::NS(Argument_get_size)( arg2 ) );

    SIXTRL_ASSERT( 0 == ::NS(Buffer_clear)( c_cpy_buffer, true ) );
    SIXTRL_ASSERT( 0 == ::NS(Buffer_reset)( c_cpy_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
        c_cpy_buffer ) == size_t{ 0 } );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) !=
                   ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) );

    status = ::NS(Argument_receive_buffer)( arg2, c_cpy_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) );

    /* --------------------------------------------------------------------- */

    ::NS(ElemByElemConfig) config_orig;

    ::NS(ElemByElemConfig_preset)( &config_orig );

    config_orig.min_particle_id = ::NS(particle_index_t){ 0 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 0 };

    ::NS(ElemByElemConfig) config_copy = config_orig;

    config_orig.min_particle_id = ::NS(particle_index_t){ 1 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 2 };

    cuda_arg_t* arg3 = ::NS(CudaArgument_new_from_raw_argument)(
        &config_orig, sizeof( config_orig ), cuda_controller );

    ASSERT_TRUE( arg3 != nullptr );

    ASSERT_TRUE( ::NS(Argument_has_argument_buffer)( arg3 ) );
    ASSERT_TRUE( ::NS(CudaArgument_has_cuda_arg_buffer)( arg3 ) );
    ASSERT_TRUE( !::NS(Argument_uses_cobjects_buffer)( arg3 ) );
    ASSERT_TRUE( ::NS(Argument_uses_raw_argument)( arg3 ) );
    ASSERT_TRUE( ::NS(Argument_get_const_ptr_raw_argument)( arg3 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_ptr_raw_argument)( arg3 ) == &config_orig );
    ASSERT_TRUE( ::NS(Argument_get_size)( arg3 ) == sizeof( config_orig ) );
    ASSERT_TRUE( ::NS(Argument_get_size)( arg3 ) <= ::NS(Argument_get_capacity)( arg3 ) );

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = ::NS(Argument_receive_raw_argument)(
        arg3, &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    config_copy.min_particle_id = ::NS(particle_index_t){ 0 };
    config_copy.max_particle_id = ::NS(particle_index_t){ 0 };

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = ::NS(Argument_send_raw_argument)(
        arg3, &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = ::NS(Argument_receive_raw_argument)(
        arg3, &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    /* --------------------------------------------------------------------- */

    cuda_arg_t* arg4 = ::NS(CudaArgument_new)( cuda_controller );

    ASSERT_TRUE( !::NS(Argument_has_argument_buffer)( arg4 ) );
    ASSERT_TRUE( !::NS(CudaArgument_has_cuda_arg_buffer)( arg4  ) );
    ASSERT_TRUE(  ::NS(CudaArgument_get_cuda_arg_buffer)( arg4 ) == nullptr );

    ASSERT_TRUE( !::NS(Argument_uses_cobjects_buffer)( arg4 ) );
    ASSERT_TRUE(  ::NS(Argument_get_const_cobjects_buffer)( arg4 ) == nullptr );

    ASSERT_TRUE( !::NS(Argument_uses_raw_argument)( arg4 ) );
    ASSERT_TRUE(  ::NS(Argument_get_const_ptr_raw_argument)( arg4 ) == nullptr );

    config_orig.min_particle_id = ::NS(particle_index_t){ 0 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 0 };

    config_copy = config_orig;

    config_orig.min_particle_id = ::NS(particle_index_t){ 1 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 2 };

    status = ::NS(Argument_send_raw_argument)(
        arg4, &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( ::NS(Argument_has_argument_buffer)( arg4 ) );
    ASSERT_TRUE( ::NS(CudaArgument_has_cuda_arg_buffer)( arg4 ) );
    ASSERT_TRUE( !::NS(Argument_uses_cobjects_buffer)( arg4 ) );
    ASSERT_TRUE( ::NS(Argument_uses_raw_argument)( arg4 ) );
    ASSERT_TRUE( ::NS(Argument_get_const_ptr_raw_argument)( arg4 ) != nullptr );
    ASSERT_TRUE( ::NS(Argument_get_const_ptr_raw_argument)( arg4 ) == &config_orig );
    ASSERT_TRUE( ::NS(Argument_get_size)( arg4 ) == sizeof( config_orig ) );
    ASSERT_TRUE( ::NS(Argument_get_size)( arg4 ) <= ::NS(Argument_get_capacity)( arg4 ) );

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = ::NS(Argument_receive_raw_argument)(
        arg4, &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    config_copy.min_particle_id = ::NS(particle_index_t){ 0 };
    config_copy.max_particle_id = ::NS(particle_index_t){ 0 };

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = ::NS(Argument_send_raw_argument)(
        arg4, &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = ::NS(Argument_receive_raw_argument)(
        arg4, &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    /* --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( c_obj_buffer );
    c_obj_buffer = nullptr;

    ::NS(Buffer_delete)( c_cpy_buffer );
    c_cpy_buffer = nullptr;

    ::NS(Controller_delete)( cuda_controller );
    cuda_controller = nullptr;

    ::NS(Argument_delete)( arg1 );
    arg1 = nullptr;

    ::NS(Argument_delete)( arg2 );
    arg2 = nullptr;

    ::NS(Argument_delete)( arg3 );
    arg3 = nullptr;

    ::NS(Argument_delete)( arg4 );
    arg4 = nullptr;
}

/* end: tests/sixtracklib/cuda/control/test_argument_cxx.cpp */

