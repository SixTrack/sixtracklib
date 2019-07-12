#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import sixtracklib as pyst
import ctypes as ct

from sixtracklib.stcommon import st_NODE_UNDEFINED_INDEX, \
    st_ARCH_STATUS_SUCCESS, st_ARCH_STATUS_GENERAL_FAILURE, \
    st_NullCudaController, st_NullCudaArgument, st_NullBuffer, \
    st_CudaArgBuffer_p, st_NullCudaArgBuffer, st_Null, \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete, \
    st_Buffer_get_size, st_Buffer_new, st_Buffer_get_data_begin_addr, \
    st_Buffer_get_data_end_addr, st_Buffer_get_objects_begin_addr, \
    st_Buffer_get_slot_size, st_Buffer_get_num_of_objects, \
    st_CudaArgument_new_from_buffer, st_CudaArgument_new, \
    st_Argument_delete, st_CudaArgument_new_from_raw_argument, \
    st_Argument_get_size, st_Argument_get_capacity, \
    st_Argument_uses_cobjects_buffer, st_Argument_get_cobjects_buffer, \
    st_Argument_get_cobjects_buffer_slot_size

from sixtracklib.stcommon import st_ElemByElemConfig, st_ElemByElemConfig_p, \
    st_NullElemByElemConfig, st_ElemByElemConfig_preset

import sixtracklib_test as testlib
from sixtracklib_test.generic_obj import GenericObj

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    num_d_values = 10
    num_e_values = 10
    num_obj = 10

    obj_buffer = CBuffer()
    for ii in range(0, num_obj):
        obj = GenericObj(cbuffer=obj_buffer, type_id=ii, a=ii, b=float(ii),
                         c=[1.0, 2.0, 3.0, 4.0],
                         num_d=num_d_values, num_e=num_e_values)

    c_obj_buffer = pyst.Buffer(cbuffer=obj_buffer)
    assert c_obj_buffer.pointer != st_NullBuffer
    assert c_obj_buffer.slot_size > 0
    assert c_obj_buffer.capacity > 0
    assert c_obj_buffer.size > 0 and c_obj_buffer.size <= c_obj_buffer.capacity
    assert c_obj_buffer.num_objects == obj_buffer.n_objects

    c_cpy_buffer = pyst.Buffer(size=c_obj_buffer.capacity)
    assert c_cpy_buffer.pointer != st_NullBuffer
    assert c_cpy_buffer.slot_size > 0
    assert c_cpy_buffer.capacity > 0
    assert c_cpy_buffer.size <= c_obj_buffer.size
    assert c_cpy_buffer.capacity >= c_obj_buffer.capacity

    ctrl = pyst.CudaController()
    assert ctrl.num_nodes > 0
    assert ctrl.has_selected_node
    assert ctrl.selected_node_index != st_NODE_UNDEFINED_INDEX.value

    arg1 = pyst.CudaArgument(ctrl=ctrl)
    assert arg1.controller.pointer != st_NullCudaController
    assert arg1.controller.pointer == ctrl.pointer
    assert not arg1.has_argument_buffer
    assert not arg1.has_cuda_arg_buffer

    assert not arg1.uses_buffer
    assert arg1.ptr_buffer == st_NullBuffer
    assert not arg1.uses_raw_argument

    arg1.send_buffer(c_obj_buffer)
    assert arg1.last_status_success
    assert arg1.has_argument_buffer
    assert arg1.has_cuda_arg_buffer
    assert arg1.cuda_arg_buffer != st_NullCudaArgBuffer
    assert arg1.uses_buffer
    assert arg1.ptr_buffer != st_NullBuffer
    assert arg1.size > 0
    assert arg1.capacity >= arg1.size
    assert arg1.size == c_obj_buffer.size

    assert not ctrl.buffer_arg_needs_remapping(arg1)
    assert not ctrl.managed_cobject_buffer_needs_remapping(
        arg1.cuda_arg_buffer, c_obj_buffer.slot_size)

    arg1.receive_buffer(c_obj_buffer)
    assert arg1.last_status_success
    assert not c_obj_buffer.needs_remapping
    assert arg1.has_argument_buffer
    assert arg1.has_cuda_arg_buffer
    assert arg1.cuda_arg_buffer != st_NullCudaArgBuffer
    assert arg1.uses_buffer
    assert arg1.ptr_buffer != st_NullBuffer
    assert arg1.size > 0
    assert arg1.capacity >= arg1.size
    assert arg1.size == c_obj_buffer.size

    assert not ctrl.buffer_arg_needs_remapping(arg1)
    assert not ctrl.managed_cobject_buffer_needs_remapping(
        arg1.cuda_arg_buffer, c_obj_buffer.slot_size)

    arg1.receive_buffer(c_cpy_buffer)
    assert arg1.last_status_success
    assert not c_cpy_buffer.needs_remapping
    assert arg1.has_argument_buffer
    assert arg1.has_cuda_arg_buffer
    assert arg1.cuda_arg_buffer != st_NullCudaArgBuffer
    assert arg1.uses_buffer
    assert arg1.ptr_buffer != st_NullBuffer
    assert arg1.size > 0
    assert arg1.capacity >= arg1.size
    assert arg1.size == c_obj_buffer.size
    assert c_cpy_buffer.num_objects == c_obj_buffer.num_objects

    # --------------------------------------------------------------------------

    arg2 = pyst.CudaArgument(buffer=c_obj_buffer, ctrl=ctrl)

    assert arg2.has_argument_buffer
    assert arg2.has_cuda_arg_buffer
    assert arg2.uses_buffer
    assert arg2.controller.pointer != st_NullCudaController
    assert arg2.controller.pointer == ctrl.pointer

    assert arg2.ptr_buffer != st_NullBuffer
    assert arg2.size == c_obj_buffer.size
    assert arg2.capacity >= arg2.size

    assert not ctrl.buffer_arg_needs_remapping(arg2)
    assert not ctrl.managed_cobject_buffer_needs_remapping(
        arg2.cuda_arg_buffer, c_obj_buffer.slot_size)

    c_cpy_buffer.clear(True).reset()
    assert c_cpy_buffer.last_status_success
    assert c_cpy_buffer.num_objects == 0

    arg2.receive_buffer(c_cpy_buffer)
    assert arg2.last_status_success
    assert not ctrl.buffer_arg_needs_remapping(arg2)
    assert c_cpy_buffer.num_objects == c_obj_buffer.num_objects

    # --------------------------------------------------------------------------

    config_orig = st_ElemByElemConfig()
    st_ElemByElemConfig_preset(ct.byref(config_orig))

    config_copy = st_ElemByElemConfig()
    st_ElemByElemConfig_preset(ct.byref(config_copy))

    config_orig.min_particle_id = 0
    config_orig.max_particle_id = 0

    config_copy.min_particle_id = 1
    config_copy.max_particle_id = 2

    assert config_orig.min_particle_id != config_copy.min_particle_id
    assert config_orig.max_particle_id != config_copy.max_particle_id

    arg3 = pyst.CudaArgument(ptr_raw_arg_begin=ct.byref(config_orig),
                             raw_arg_size=ct.sizeof(config_orig), ctrl=ctrl)

    assert arg3.has_argument_buffer
    assert arg3.has_cuda_arg_buffer
    assert not arg3.uses_buffer
    assert arg3.uses_raw_argument
    assert arg3.ptr_raw_argument != st_Null
    assert arg3.size > 0
    assert arg3.size <= arg3.capacity
    assert arg3.size == ct.sizeof(config_orig)
    assert arg3.controller.pointer != st_NullCudaController
    assert arg3.controller.pointer == ctrl.pointer

    arg3.receive_raw_argument(
        ct.byref(config_copy), ct.sizeof(config_copy))

    assert arg3.last_status_success
    assert config_orig.min_particle_id == config_copy.min_particle_id
    assert config_orig.max_particle_id == config_copy.max_particle_id

    config_orig.min_particle_id = 1
    config_orig.max_particle_id = 2

    assert config_orig.min_particle_id != config_copy.min_particle_id
    assert config_orig.max_particle_id != config_copy.max_particle_id

    arg3.send_raw_argument(ct.byref(config_copy), ct.sizeof(config_copy))

    assert arg3.last_status_success

    arg3.receive_raw_argument(
        ct.byref(config_orig), ct.sizeof(config_orig))

    assert arg3.last_status_success
    assert config_orig.min_particle_id == config_copy.min_particle_id
    assert config_orig.max_particle_id == config_copy.max_particle_id

    # --------------------------------------------------------------------------

    arg4 = pyst.CudaArgument(ctrl=ctrl)
    assert arg4.controller.pointer != st_NullCudaController
    assert arg4.controller.pointer == ctrl.pointer
    assert not arg4.has_argument_buffer
    assert not arg4.has_cuda_arg_buffer

    assert not arg4.uses_buffer
    assert arg4.ptr_buffer == st_NullBuffer
    assert not arg4.uses_raw_argument

    config_orig.min_particle_id = 1
    config_orig.max_particle_id = 2

    assert config_orig.min_particle_id != config_copy.min_particle_id
    assert config_orig.max_particle_id != config_copy.max_particle_id

    arg4.send_raw_argument(ct.byref(config_orig), ct.sizeof(config_orig))
    assert arg4.last_status_success

    assert arg4.has_argument_buffer
    assert arg4.has_cuda_arg_buffer
    assert not arg4.uses_buffer
    assert arg4.uses_raw_argument
    assert arg4.ptr_raw_argument != st_Null
    assert arg4.size > 0
    assert arg4.size <= arg4.capacity
    assert arg4.size == ct.sizeof(config_orig)
    assert arg4.controller.pointer != st_NullCudaController
    assert arg4.controller.pointer == ctrl.pointer

    arg4.receive_raw_argument(
        ct.byref(config_copy), ct.sizeof(config_copy))

    assert arg4.last_status_success
    assert config_orig.min_particle_id == config_copy.min_particle_id
    assert config_orig.max_particle_id == config_copy.max_particle_id

    config_orig.min_particle_id = 0
    config_orig.max_particle_id = 0

    assert config_orig.min_particle_id != config_copy.min_particle_id
    assert config_orig.max_particle_id != config_copy.max_particle_id

    arg4.send_raw_argument(ct.byref(config_copy), ct.sizeof(config_copy))

    assert arg4.last_status_success

    arg4.receive_raw_argument(
        ct.byref(config_orig), ct.sizeof(config_orig))

    assert arg4.last_status_success
    assert config_orig.min_particle_id == config_copy.min_particle_id
    assert config_orig.max_particle_id == config_copy.max_particle_id

    sys.exit(0)
