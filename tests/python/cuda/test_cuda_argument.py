#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import pysixtracklib as pyst
from pysixtracklib import stcommon as st
import pysixtracklib_test as testlib
from pysixtracklib_test.generic_obj import GenericObj
import ctypes as ct

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    num_d_values = 10
    num_e_values = 10
    num_obj      = 10

    CBuffer obj_buffer
    for ii in range( 0, num_obj ):
        obj = GenericObj( cbuffer=obj_buffer, type_id=ii, a=ii, b=float(ii),
            c = [ 1.0, 2.0, 3.0, 4.0 ],
            num_d=num_d_values, num_e=num_e_values )

    c_obj_buffer = st.st_Buffer_new_mapped_on_cbuffer( obj_buffer )
    assert c_obj_buffer != st.st_NullBuffer

    slot_size = st.st_Buffer_get_slot_size( c_obj_buffer )
    assert slot_size > 0

    c_cpy_buffer = st.st_Buffer_new( st.st_Buffer_get_capacity( c_obj_buffer ) )
    assert c_cpy_buffer != st.st_NullBuffer

    cuda_controller = st.st_CudaController_create()
    assert cuda_controller != st_NullCudaController
    assert st.st_Controller_has_selected_node( cuda_controller )

    arg1 = st.st_CudaArgument_new( cuda_controller )

    assert arg1 != st.st_NullBuffer
    assert not st.st_Argument_has_argument_buffer( arg1 )
    assert not st.st_Argument_has_cuda_arg_buffer( arg1 )
    assert st.st_CudaArgument_get_cuda_arg_buffer( arg1 ) == \
        st.st_null_cuda_arg_buffer_t

    assert not st.st_Argument_uses_cobjects_buffer( arg1 )
    assert  st.st_Argument_get_const_cobjects_buffer( arg1 ) == st_NullBuffer

    assert not st.st_Argument_uses_raw_argument( arg1 )
    assert  st.st_Argument_get_const_ptr_raw_argument( arg1 ) == st.st_Null

    status = st.st_Argument_send_buffer( arg1, c_obj_buffer )

    assert status == st.st_ARCH_STATUS_SUCCESS
    assert st.st_Argument_has_argument_buffer( arg1 )
    assert st.st_Argument_has_cuda_arg_buffer( arg1 )
    assert st.st_CudaArgument_get_cuda_arg_buffer( arg1 ) != \
        st_null_cuda_arg_buffer_t

    assert st.st_Argument_uses_cobjects_buffer( arg1 )
    assert st.st_Argument_get_const_cobjects_buffer( arg1 ) != st_NullCudaArgument
    assert st.st_Argument_get_const_cobjects_buffer( arg1 ) == c_obj_buffer


    assert st.st_Argument_get_size)( arg1 ) > 0
    assert st.st_Argument_get_capacity( arg1 ) >= st.st_Argument_get_size( arg1 )
    assert st.st_Argument_get_size)( arg1 ) == st.st_Buffer_get_size( c_obj_buffer )
    assert st.st_CudaController_is_managed_cobject_buffer_remapped(
        cuda_controller, st.st_CudaArgument_get_cuda_arg_buffer( arg1 ),
            slot_size )

    status = st.st_Argument_receive_buffer)( arg1, c_obj_buffer )

    assert status == st.st_ARCH_STATUS_SUCCESS

    assert st.st_Argument_has_argument_buffer)( arg1 )
    assert st.st_Argument_has_cuda_arg_buffer)( arg1 )
    assert st.st_CudaArgument_get_cuda_arg_buffer)( arg1 ) != st_NullCudaArgument

    assert st.st_Argument_uses_cobjects_buffer)( arg1 )
    assert st.st_Argument_get_const_cobjects_buffer)( arg1 ) != st_NullCudaArgument
    assert st.st_Argument_get_const_cobjects_buffer)( arg1 ) == c_obj_buffer

    assert not st.st_Buffer_needs_remapping( c_obj_buffer )
    assert st.st_CudaController_is_managed_cobject_buffer_remapped(
        cuda_controller, st.st_CudaArgument_get_cuda_arg_buffer(
            arg1 ), slot_size )

    status = st.st_Argument_receive_buffer( arg1, c_cpy_buffer )

    assert status == st.st_ARCH_STATUS_SUCCESS
    assert st.st_Argument_uses_cobjects_buffer( arg1 )
    assert st.st_Argument_get_const_cobjects_buffer( arg1 ) != st_NullBuffer
    assert st.st_Argument_get_const_cobjects_buffer( arg1 ) == c_obj_buffer


    assert st.st_CudaController_is_managed_cobject_buffer_remapped(
        cuda_controller, st.st_CudaArgument_get_cuda_arg_buffer( arg1 ),
            slot_size )

    assert !st.st_Buffer_needs_remapping( c_cpy_buffer )
    assert !st.st_Buffer_needs_remapping( c_obj_buffer )

    assert st.st_(Buffer_get_size( c_cpy_buffer ) == \
           st.st_Buffer_get_size( c_obj_buffer )

    assert st.st_Buffer_get_num_of_objects( c_cpy_buffer ) == \
           st.st_Buffer_get_num_of_objects( c_obj_buffer )

    #* ---------------------------------------------------------------------

    cuda_arg_t* arg2 = st.st_CudaArgument_new_from_buffer)( c_obj_buffer, cuda_controller);

    assert arg2 != st_NullCudaArgument

    assert st.st_Argument_has_argument_buffer)( arg2 )
    assert st.st_Argument_has_cuda_arg_buffer)( arg2 )
    assert st.st_Argument_uses_cobjects_buffer)( arg2 )

    assert st.st_Argument_get_const_cobjects_buffer)( arg2 ) != st_NullCudaArgument
    assert st.st_Argument_get_const_cobjects_buffer)( arg2 ) == c_obj_buffer
    assert st.st_Argument_get_size)( arg2 ) == ::NS(Buffer_get_size)( c_obj_buffer )
    assert st.st_Argument_get_capacity)( arg2 ) >= st.st_Argument_get_size)( arg2 )

    SIXTRL_ASSERT( 0 == ::NS(Buffer_clear)( c_cpy_buffer, true ) );
    SIXTRL_ASSERT( 0 == ::NS(Buffer_reset)( c_cpy_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) == size_t{ 0 } );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( c_obj_buffer ) != ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) );

    status = st.st_Argument_receive_buffer)( arg2, c_cpy_buffer );

    assert status == st::ARCH_STATUS_SUCCESS

    assert !::NS(Buffer_needs_remapping)( c_cpy_buffer )

    assert ::NS(Buffer_get_num_of_objects)( c_cpy_buffer ) == ::NS(Buffer_get_num_of_objects)( c_obj_buffer )

    sys.exit(0)
