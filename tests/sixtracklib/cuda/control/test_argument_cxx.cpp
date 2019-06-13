#include "sixtracklib/cuda/argument.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/cuda/controller.hpp"

TEST( CXX_CudaArgumentTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_arg_t    = st::CudaArgument;
    using size_t        = cuda_arg_t::size_type;
    using status_t      = cuda_arg_t::status_t;
    using cuda_ctrl_t   = cuda_arg_t::cuda_controller_t;
    using buffer_t      = cuda_arg_t::buffer_t;
    using c_buffer_t    = cuda_arg_t::c_buffer_t;
    using generic_obj_t = ::NS(GenericObj);
    using type_id_t     = ::NS(object_type_id_t);

    /* --------------------------------------------------------------------- */
    /* Prepare the cobject buffer's (both C++ and C99) */

    size_t constexpr num_d_values = size_t{ 10 };
    size_t constexpr num_e_values = size_t{ 10 };
    size_t constexpr num_obj      = size_t{ 10 };

    c_buffer_t* c_obj_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    buffer_t obj_buffer;

    for( size_t ii = size_t{ 0 } ; ii < num_obj ; ++ii )
    {
        type_id_t const type_id = static_cast< type_id_t >( ii );

        generic_obj_t* ptr_c_obj = ::NS(GenericObj_new)(
            c_obj_buffer, type_id, num_d_values, num_e_values );

        generic_obj_t* ptr_cxx_obj = ::NS(GenericObj_new)(
            obj_buffer.getCApiPtr(), type_id, num_d_values, num_e_values );

        SIXTRL_ASSERT( ptr_c_obj != nullptr );
        SIXTRL_ASSERT( ptr_cxx_obj != nullptr );
        SIXTRL_ASSERT( ptr_cxx_obj != ptr_c_obj );
    }

    size_t const slot_size = ::NS(Buffer_get_slot_size)( c_obj_buffer );
    SIXTRL_ASSERT( slot_size > size_t{ 0 } );
    SIXTRL_ASSERT( slot_size == obj_buffer.getSlotSize() );

    c_buffer_t* c_cpy_buffer = ::NS(Buffer_new)(
        ::NS(Buffer_get_capacity( c_obj_buffer ) ) );

    SIXTRL_ASSERT( c_cpy_buffer != nullptr );

    SIXTRL_ASSERT( ::NS(Buffer_get_size)( c_cpy_buffer ) <=
                   ::NS(Buffer_get_size)( c_obj_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_capacity)( c_cpy_buffer ) >=
                   ::NS(Buffer_get_capacity)( c_obj_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
        c_cpy_buffer ) == size_t{ 0 } );

    SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );


    /* --------------------------------------------------------------------- */

    cuda_ctrl_t cuda_controller;

    cuda_arg_t arg1(  &cuda_controller );
    ASSERT_TRUE( arg1.ptrControllerBase() != nullptr );
    ASSERT_TRUE( arg1.cudaController() == &cuda_controller );
    ASSERT_TRUE( !arg1.hasArgumentBuffer() );
    ASSERT_TRUE( !arg1.hasCudaArgBuffer() );
    ASSERT_TRUE(  arg1.cudaArgBuffer() == nullptr );

    ASSERT_TRUE( !arg1.usesCObjectsBuffer() );
    ASSERT_TRUE(  arg1.ptrCObjectsBuffer() == nullptr );

    ASSERT_TRUE( !arg1.usesCObjectsCxxBuffer() );
    ASSERT_TRUE(  arg1.ptrCObjectsCxxBuffer() == nullptr );

    ASSERT_TRUE( !arg1.usesRawArgument() );
    ASSERT_TRUE(  arg1.ptrRawArgument() == nullptr );

    status_t status = arg1.send( c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( arg1.hasArgumentBuffer() );
    ASSERT_TRUE( arg1.hasCudaArgBuffer() );
    ASSERT_TRUE( arg1.cudaArgBuffer() != nullptr );

    ASSERT_TRUE( arg1.usesCObjectsBuffer() );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() == c_obj_buffer );

    ASSERT_TRUE( !arg1.usesCObjectsCxxBuffer() );
    ASSERT_TRUE(  arg1.ptrCObjectsCxxBuffer() == nullptr );

    ASSERT_TRUE(  arg1.size() > size_t{ 0 } );
    ASSERT_TRUE(  arg1.capacity() >= arg1.size() );
    ASSERT_TRUE(  arg1.size() == ::NS(Buffer_get_size)( c_obj_buffer ) );

    ASSERT_TRUE(  cuda_controller.isRemapped( arg1 ) );

    ASSERT_TRUE(  cuda_controller.isRemapped(
        arg1.cudaArgBuffer(), slot_size ) );

    status = arg1.receive( c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( arg1.hasArgumentBuffer() );
    ASSERT_TRUE( arg1.hasCudaArgBuffer() );
    ASSERT_TRUE( arg1.cudaArgBuffer() != nullptr );

    ASSERT_TRUE( arg1.usesCObjectsBuffer() );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() == c_obj_buffer );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );
    ASSERT_TRUE(  cuda_controller.isRemapped( arg1 ) );
    ASSERT_TRUE(  cuda_controller.isRemapped(
        arg1.cudaArgBuffer(), slot_size ) );

    status = arg1.receive( c_cpy_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( arg1.usesCObjectsBuffer() );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg1.ptrCObjectsBuffer() == c_obj_buffer );

    ASSERT_TRUE( cuda_controller.isRemapped( arg1 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg1.cudaArgBuffer(), slot_size ) );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );
    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_size)( c_obj_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) );

    /* --------------------------------------------------------------------- */

    cuda_arg_t arg2( c_obj_buffer, &cuda_controller );

    ASSERT_TRUE( arg2.hasArgumentBuffer() );
    ASSERT_TRUE( arg2.hasCudaArgBuffer() );
    ASSERT_TRUE( arg2.usesCObjectsBuffer() );
    ASSERT_TRUE( arg2.ptrControllerBase() != nullptr );
    ASSERT_TRUE( arg2.cudaController() == &cuda_controller );

    ASSERT_TRUE( arg2.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg2.ptrCObjectsBuffer() == c_obj_buffer );
    ASSERT_TRUE( arg2.size() == ::NS(Buffer_get_size)( c_obj_buffer ) );
    ASSERT_TRUE( arg2.capacity() >= arg2.size() );

    ASSERT_TRUE( cuda_controller.isRemapped( arg2 ) );

    SIXTRL_ASSERT( 0 == ::NS(Buffer_clear)( c_cpy_buffer, true ) );
    SIXTRL_ASSERT( 0 == ::NS(Buffer_reset)( c_cpy_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
        c_cpy_buffer ) == size_t{ 0 } );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) !=
                   ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) );


    status = arg2.receive( c_cpy_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( cuda_controller.isRemapped( arg2 ) );
    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) );

    /* --------------------------------------------------------------------- */

    cuda_arg_t arg3( obj_buffer, &cuda_controller );

    ASSERT_TRUE( arg3.hasArgumentBuffer() );
    ASSERT_TRUE( arg3.hasCudaArgBuffer() );
    ASSERT_TRUE( arg3.usesCObjectsCxxBuffer() );

    ASSERT_TRUE( arg3.ptrCObjectsCxxBuffer() != nullptr );
    ASSERT_TRUE( arg3.ptrCObjectsCxxBuffer() == &obj_buffer );

    ASSERT_TRUE( arg3.usesCObjectsBuffer() );
    ASSERT_TRUE( arg3.ptrControllerBase() != nullptr );
    ASSERT_TRUE( arg3.cudaController() == &cuda_controller );

    ASSERT_TRUE( cuda_controller.isRemapped( arg3 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg3.cudaArgBuffer(), slot_size ) );

    obj_buffer.clear( true );
    obj_buffer.reset();

    ASSERT_TRUE( obj_buffer.getNumObjects() == size_t{ 0 } );

    status = arg3.send( obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( cuda_controller.isRemapped( arg3 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg3.cudaArgBuffer(), slot_size ) );

    status = arg3.send( c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( cuda_controller.isRemapped( arg3 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg3.cudaArgBuffer(), slot_size ) );

    status = arg3.receive( obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( cuda_controller.isRemapped( arg3 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg3.cudaArgBuffer(), slot_size ) );

    ASSERT_TRUE( !obj_buffer.needsRemapping() );
    ASSERT_TRUE( obj_buffer.getNumObjects() > size_t{ 0 } );
    ASSERT_TRUE( obj_buffer.getNumObjects() ==
                 ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) );

    /* --------------------------------------------------------------------- */

    cuda_arg_t arg4( &cuda_controller );

    ASSERT_TRUE(  arg4.ptrControllerBase() != nullptr );
    ASSERT_TRUE(  arg4.cudaController() == &cuda_controller );
    ASSERT_TRUE( !arg4.hasArgumentBuffer() );
    ASSERT_TRUE( !arg4.hasCudaArgBuffer() );
    ASSERT_TRUE(  arg4.cudaArgBuffer() == nullptr );

    ASSERT_TRUE( !arg4.usesCObjectsBuffer() );
    ASSERT_TRUE(  arg4.ptrCObjectsBuffer() == nullptr );

    ASSERT_TRUE( !arg4.usesCObjectsCxxBuffer() );
    ASSERT_TRUE(  arg4.ptrCObjectsCxxBuffer() == nullptr );

    ASSERT_TRUE( !arg4.usesRawArgument() );
    ASSERT_TRUE(  arg4.ptrRawArgument() == nullptr );

    status = arg4.send( obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( arg4.hasArgumentBuffer() );
    ASSERT_TRUE( arg4.hasCudaArgBuffer() );
    ASSERT_TRUE( arg4.cudaArgBuffer() != nullptr );

    ASSERT_TRUE( arg4.usesCObjectsBuffer() );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() == obj_buffer.getCApiPtr() );
    ASSERT_TRUE( arg4.usesCObjectsCxxBuffer() );
    ASSERT_TRUE( arg4.ptrCObjectsCxxBuffer() == &obj_buffer );

    ASSERT_TRUE( arg4.size() > size_t{ 0 } );
    ASSERT_TRUE( arg4.capacity() >= arg1.size() );
    ASSERT_TRUE( arg4.size() == obj_buffer.getSize() );

    ASSERT_TRUE( cuda_controller.isRemapped( arg4 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg4.cudaArgBuffer(), slot_size ) );

    status = arg4.receive( c_obj_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( arg4.hasArgumentBuffer() );
    ASSERT_TRUE(  cuda_controller.isRemapped( arg1 ) );
    ASSERT_TRUE( arg4.hasCudaArgBuffer() );
    ASSERT_TRUE( arg4.cudaArgBuffer() != nullptr );

    ASSERT_TRUE( arg4.usesCObjectsBuffer() );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() == obj_buffer.getCApiPtr() );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );
    ASSERT_TRUE(  cuda_controller.isRemapped(
        arg4.cudaArgBuffer(), slot_size ) );

    status = arg4.receive( c_cpy_buffer );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );
    ASSERT_TRUE( arg4.usesCObjectsBuffer() );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() != nullptr );
    ASSERT_TRUE( arg4.ptrCObjectsBuffer() == obj_buffer.getCApiPtr() );

    ASSERT_TRUE( cuda_controller.isRemapped( arg4 ) );
    ASSERT_TRUE( cuda_controller.isRemapped(
        arg4.cudaArgBuffer(), slot_size ) );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_cpy_buffer ) );
    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( c_obj_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( c_cpy_buffer ) ==
                 ::NS(Buffer_get_size)( c_obj_buffer ) );

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

    cuda_arg_t arg5( &config_orig, sizeof( config_orig ), &cuda_controller );

    ASSERT_TRUE( arg5.hasArgumentBuffer() );
    ASSERT_TRUE( arg5.hasCudaArgBuffer() );
    ASSERT_TRUE( !arg5.usesCObjectsCxxBuffer() );
    ASSERT_TRUE( !arg5.usesCObjectsBuffer() );
    ASSERT_TRUE( arg5.usesRawArgument() );
    ASSERT_TRUE( arg5.ptrRawArgument() != nullptr );
    ASSERT_TRUE( arg5.ptrRawArgument() == &config_orig );
    ASSERT_TRUE( arg5.size() == sizeof( config_orig ) );
    ASSERT_TRUE( arg5.size() <= arg5.capacity() );

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = arg5.receive( &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    config_copy.min_particle_id = ::NS(particle_index_t){ 0 };
    config_copy.max_particle_id = ::NS(particle_index_t){ 0 };

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = arg5.send( &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = arg5.receive( &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    /* --------------------------------------------------------------------- */

    cuda_arg_t arg6( &cuda_controller );

    ASSERT_TRUE(  arg6.ptrControllerBase() != nullptr );
    ASSERT_TRUE(  arg6.cudaController() == &cuda_controller );
    ASSERT_TRUE( !arg6.hasArgumentBuffer() );
    ASSERT_TRUE( !arg6.hasCudaArgBuffer() );
    ASSERT_TRUE(  arg6.cudaArgBuffer() == nullptr );

    ASSERT_TRUE( !arg6.usesCObjectsBuffer() );
    ASSERT_TRUE(  arg6.ptrCObjectsBuffer() == nullptr );

    ASSERT_TRUE( !arg6.usesCObjectsCxxBuffer() );
    ASSERT_TRUE(  arg6.ptrCObjectsCxxBuffer() == nullptr );

    ASSERT_TRUE( !arg6.usesRawArgument() );
    ASSERT_TRUE(  arg6.ptrRawArgument() == nullptr );


    config_orig.min_particle_id = ::NS(particle_index_t){ 0 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 0 };

    config_copy = config_orig;

    config_orig.min_particle_id = ::NS(particle_index_t){ 1 };
    config_orig.max_particle_id = ::NS(particle_index_t){ 2 };

    status = arg6.send( &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( arg6.hasArgumentBuffer() );
    ASSERT_TRUE( arg6.hasCudaArgBuffer() );
    ASSERT_TRUE( !arg6.usesCObjectsCxxBuffer() );
    ASSERT_TRUE( !arg6.usesCObjectsBuffer() );
    ASSERT_TRUE( arg6.usesRawArgument() );
    ASSERT_TRUE( arg6.ptrRawArgument() != nullptr );
    ASSERT_TRUE( arg6.ptrRawArgument() == &config_orig );
    ASSERT_TRUE( arg6.size() == sizeof( config_orig ) );
    ASSERT_TRUE( arg6.size() <= arg6.capacity() );

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = arg6.receive( &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    config_copy.min_particle_id = ::NS(particle_index_t){ 0 };
    config_copy.max_particle_id = ::NS(particle_index_t){ 0 };

    SIXTRL_ASSERT( config_orig.min_particle_id != config_copy.min_particle_id );
    SIXTRL_ASSERT( config_orig.max_particle_id != config_copy.max_particle_id );

    status = arg6.send( &config_copy, sizeof( config_copy ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = arg6.receive( &config_orig, sizeof( config_orig ) );

    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    ASSERT_TRUE( config_orig.min_particle_id == config_copy.min_particle_id );
    ASSERT_TRUE( config_orig.max_particle_id == config_copy.max_particle_id );

    /* --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( c_obj_buffer );
    c_obj_buffer = nullptr;

    ::NS(Buffer_delete)( c_cpy_buffer );
    c_cpy_buffer = nullptr;
}

/* end: tests/sixtracklib/cuda/control/test_argument_cxx.cpp */



