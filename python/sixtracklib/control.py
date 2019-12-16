#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .stcommon import st_Controller_delete, st_Controller_clear, \
    st_Controller_get_arch_id, st_Controller_uses_nodes, \
    st_Controller_has_arch_string, st_Controller_get_arch_string, \
    st_Controller_has_config_string, st_Controller_get_config_string, \
    st_Controller_send_detailed, st_Controller_send_buffer, \
    st_Controller_receive_detailed, st_Controller_receive_buffer, \
    st_Controller_is_cobjects_buffer_arg_remapped, \
    st_Controller_remap_cobjects_buffer_arg, st_Controller_is_ready_to_send, \
    st_Controller_is_ready_to_remap, st_Controller_is_ready_to_run_kernel, \
    st_Controller_is_ready_to_receive, st_Controller_is_in_debug_mode, \
    st_Controller_enable_debug_mode, st_Controller_disable_debug_mode, \
    st_Controller_get_num_of_kernels, st_Controller_get_kernel_work_items_dim, \
    st_Controller_get_kernel_work_groups_dim, st_Controller_kernel_has_name, \
    st_Controller_get_num_of_kernel_arguments, st_Controller_has_kernel_id, \
    st_Controller_get_kernel_name_string, st_Controller_has_kernel_by_name, \
    st_Controller_get_ptr_kernel_config_base, \
    st_Controller_get_ptr_kernel_config_base_by_name, \
    st_Controller_has_remap_cobject_buffer_kernel, \
    st_Controller_get_remap_cobject_buffer_kernel_id, \
    st_Controller_set_remap_cobject_buffer_kernel_id, \
    st_Controller_has_remap_cobject_buffer_debug_kernel, \
    st_Controller_get_remap_cobject_buffer_debug_kernel_id, \
    st_Controller_set_remap_cobject_buffer_debug_kernel_id, \
    st_Controller_get_num_available_nodes, \
    st_Controller_get_available_node_indices, \
    st_Controller_is_node_available_by_index, \
    st_Controller_get_ptr_node_id_by_index, \
    st_Controller_get_ptr_node_info_base_by_index, \
    st_Controller_get_min_available_node_index, \
    st_Controller_get_max_available_node_index, \
    st_Controller_has_default_node, st_Controller_get_default_node_id, \
    st_Controller_get_default_node_index, \
    st_Controller_get_default_node_info_base, \
    st_Controller_has_selected_node, \
    st_Controller_get_ptr_selected_node_id, \
    st_Controller_get_selected_node_index, \
    st_Controller_get_ptr_selected_node_info_base, \
    st_Controller_can_change_selected_node, \
    st_Controller_can_directly_change_selected_node, \
    st_Controller_can_unselect_node, st_Controller_is_default_node_id, \
    st_Controller_is_default_node, st_Controller_is_default_node_index, \
    st_Controller_is_default_platform_id_and_device_id, \
    st_Controller_get_ptr_node_info_base, \
    st_Controller_get_ptr_node_info_base_by_node_id, \
    st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id, \
    st_Controller_get_node_index_by_node_id, st_Controller_get_node_index, \
    st_Controller_get_node_index_by_platform_id_and_device_id, \
    st_Controller_is_node_available_by_node_id, \
    st_Controller_is_node_available, \
    st_Controller_is_node_available_by_platform_id_and_device_id, \
    st_Controller_select_node_by_index, st_Controller_select_node_by_node_id, \
    st_Controller_select_node, \
    st_Controller_select_node_by_plaform_id_and_device_id, \
    st_Controller_change_selected_node, st_Controller_unselect_node, \
    st_Controller_unselect_node_by_index, \
    st_Controller_unselect_node_by_node_id, \
    st_Controller_unselect_node_by_platform_id_and_device_id



from .stcommon import st_Argument_delete, \
    st_Argument_get_arch_id, st_Argument_has_arch_string, \
    st_Argument_get_arch_string, st_Argument_send_again, \
    st_Argument_send_buffer, st_Argument_send_buffer_without_remap, \
    st_Argument_send_raw_argument, \
    st_Argument_receive_again, st_Argument_receive_buffer, \
    st_Argument_receive_buffer_without_remap, \
    st_Argument_receive_raw_argument, st_Argument_remap_cobjects_buffer, \
    st_Argument_uses_cobjects_buffer, st_Argument_get_cobjects_buffer, \
    st_Argument_get_cobjects_buffer_slot_size, \
    st_Argument_uses_raw_argument, st_Argument_get_const_ptr_raw_argument, \
    st_Argument_get_ptr_raw_argument, st_Argument_get_size, \
    st_Argument_get_capacity, st_Argument_has_argument_buffer, \
    st_Argument_requires_argument_buffer, \
    st_Argument_get_ptr_base_controller
from .stcommon import \
    st_NodeInfoBase_p, st_NullNodeInfoBase, st_NodeInfo_delete, \
    st_NodeInfo_get_ptr_const_node_id, st_NodeInfo_get_platform_id, \
    st_NodeInfo_get_device_id, st_NodeInfo_has_node_index, \
    st_NodeInfo_get_node_index, st_NodeInfo_is_default_node, \
    st_NodeInfo_is_selected_node, st_NodeInfo_get_arch_id, \
    st_NodeInfo_has_arch_string, st_NodeInfo_get_arch_string, \
    st_NodeInfo_has_platform_name, st_NodeInfo_get_platform_name, \
    st_NodeInfo_has_device_name, st_NodeInfo_get_device_name, \
    st_NodeInfo_has_description, st_NodeInfo_get_description, \
    st_NodeInfo_print_out, st_NodeInfo_get_required_output_str_length, \
    _st_NodeInfo_convert_to_string
import ctypes as ct
import cobjects
from cobjects import CBuffer
from .buffer import Buffer

from .stcommon import st_Null, st_NullChar, st_Buffer_p, st_NullBuffer, \
    st_ARCH_STATUS_GENERAL_FAILURE, st_ARCH_STATUS_SUCCESS, \
    st_NODE_UNDEFINED_INDEX, st_NODE_ILLEGAL_PLATFORM_ID, \
    st_NODE_ILLEGAL_DEVICE_ID, st_node_platform_id_t, st_node_device_id_t, \
    st_kernel_id_t, st_node_index_t, st_arch_status_t, st_buffer_size_t, \
    st_arch_size_t, st_NodeId_p, st_NullNodeId, \
    st_ArgumentBase_p, st_NullArgumentBase, \
    st_ControllerBase_p, st_NullControllerBase

from .stcommon import st_NodeId_create, st_NodeId_new, \
    st_NodeId_new_from_string, st_NodeId_new_detailed, st_NodeId_delete, \
    st_NodeId_is_valid, st_NodeId_get_platform_id, st_NodeId_get_device_id, \
    st_NodeId_has_node_index, st_NodeId_get_node_index, _st_NodeId_to_string


def raise_error_if_status_not_success(
        status, msg=None, cls=RuntimeError, prev=None):
    if (prev is None or prev != status) and \
            status != st_ARCH_STATUS_SUCCESS.value:
        if msg is None:
            msg = "an error occured; status:{0}".format(status)
        raise cls(msg)


class NodeId(object):
    def __init__(self, node_id_str=None, platform_id=None, device_id=None,
                 node_index=None, ext_ptr_node_id=st_NullNodeId, orig=None):
        self._ptr_node_id = st_NullNodeId
        self._owns_ptr = True

        if ext_ptr_node_id != st_NullNodeId:
            self._ptr_node_id = ext_ptr_node_id
            self._owns_ptr = False
        elif node_id_str is not None:
            node_id_str = node_id_str.strip().encode('utf-8')
            _node_id_str = ct.c_char_p(node_id_str)
            self._ptr_node_id = st_NodeId_new_from_string(_node_id_str)
            del _node_id_str
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            if node_index is None:
                self._ptr_node_id = st_NodeId_new(
                    st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))
            elif node_index is not None and \
                    node_index != st_NODE_UNDEFINED_INDEX.value:
                self._ptr_node_id = st_NodeId_new_detailed(
                    st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id),
                    st_node_index_t(node_index))
        elif orig is not None and isinstance(orig, NodeId):
            self._ptr_node_id = st_NodeId_new_detailed(
                st_node_platform_id_t(orig.platform_id),
                st_node_device_id_t(orig.device_id),
                st_node_index_t(orig.node_index))

        if self._ptr_node_id == st_NullNodeId:
            raise ValueError("Unable to create NodeId object")

    def __del__(self):
        if self._ptr_node_id != st_NullNodeId and self._owns_ptr:
            st_NodeId_delete(self._ptr_node_id)
            self._owns_ptr = False
        self._ptr_node_id = st_NullNodeId

    @staticmethod
    def to_string(node_id):
        if isinstance(node_id, NodeId):
            _ptr_node_id = node_id.pointer
        else:
            _ptr_node_id = node_id

        _max_str_len = ct.c_uint64(64)
        _int_str = ct.create_string_buffer(_max_str_len.value)
        _status = _st_NodeId_to_string(_ptr_node_id, _int_str, _max_str_len)
        node_id_str = None
        if _status == st_ARCH_STATUS_SUCCESS.value:
            node_id_str = bytes(_int_str.value).decode('utf-8')
        return node_id_str

    @property
    def pointer(self):
        return self._ptr_node_id

    @property
    def is_valid(self):
        return self._ptr_node_id != st_NullNodeId and \
            st_NodeId_is_valid(self._ptr_node_id)

    @property
    def platform_id(self):
        if self._ptr_node_id == st_NullNodeId:
            raise ValueError("Attempting to read from uninitialized NodeId")
        return st_NodeId_get_platform_id(self._ptr_node_id)

    @property
    def device_id(self):
        if self._ptr_node_id == st_NullNodeId:
            raise ValueError("Attempting to read from uninitialized NodeId")
        return st_NodeId_get_device_id(self._ptr_node_id)

    @property
    def has_node_index(self):
        if self._ptr_node_id == st_NullNodeId:
            raise ValueError("Attempting to read from uninitialized NodeId")
        return st_NodeId_has_node_index(self._ptr_node_id)

    @property
    def node_index(self):
        if self._ptr_node_id == st_NullNodeId:
            raise ValueError("Attempting to read from uninitialized NodeId")
        return st_NodeId_get_node_index(self._ptr_node_id)

    @property
    def node_id_str(self):
        return NodeId.to_string(self.pointer)

    def __str__(self):
        return NodeId.to_string(self.pointer)


class NodeInfoBase(object):
    def __init__(self, ptr_node_info=st_NullNodeInfoBase, owns_ptr=True):
        if ptr_node_info != st_NullNodeInfoBase:
            self._ptr_node_info = ptr_node_info
            self._owns_ptr = owns_ptr
        else:
            self._ptr_node_info = st_NullNodeInfoBase
            self._owns_ptr = False
        self._last_status = st_ARCH_STATUS_SUCCESS.value

    def __del__(self):
        if self._ptr_node_info != st_NullNodeInfoBase and self._owns_ptr:
            st_NodeInfo_delete(self._ptr_node_info)

    @staticmethod
    def to_string(node_info):
        if isinstance(node_info, NodeInfoBase):
            _ptr_node_info = node_info.pointer
        else:
            _ptr_node_info = node_info
        str_repr = None
        if _ptr_node_info != st_NullNodeInfoBase:
            _requ_str_capacity = 1 + \
                st_NodeInfo_get_required_output_str_length(_ptr_node_info)
            _max_str_capacity = ct.c_uint64(_requ_str_capacity)
            _temp_str = ct.create_string_buffer(_requ_str_capacity)
            _status = _st_NodeInfo_convert_to_string(
                _ptr_node_info, _max_str_capacity, _temp_str)
            if _status == st_ARCH_STATUS_SUCCESS.value:
                str_repr = bytes(_temp_str.value).decode('utf-8')
        return str_repr

    @property
    def pointer(self):
        return self._ptr_node_info

    @property
    def last_status(self):
        return self._last_status

    @property
    def last_status_success(self):
        return self._last_status == st_ARCH_STATUS_SUCCESS.value

    @property
    def arch_id(self):
        return st_NodeInfo_get_arch_id(self._ptr_node_info)

    @property
    def arch_str(self):
        arch_str = None
        if st_NodeInfo_has_arch_string(self._ptr_node_info):
            arch_str = bytes(st_NodeInfo_get_arch_string(
                self._ptr_node_info)).decode('utf-8')
        return arch_str

    @property
    def node_id(self):
        return NodeId(ext_ptr_node_id=st_NodeInfo_get_ptr_const_node_id(
            self._ptr_node_info))

    @property
    def platform_id(self):
        return st_NodeInfo_get_platform_id(self._ptr_node_info)

    @property
    def device_id(self):
        return st_NodeInfo_get_device_id(self._ptr_node_info)

    @property
    def has_node_index(self):
        return st_NodeInfo_has_node_index(self._ptr_node_info)

    @property
    def node_index(self):
        return st_NodeInfo_get_node_index(self._ptr_node_info)

    @property
    def has_platform_name(self):
        return st_NodeInfo_has_platform_name(self._ptr_node_info)

    @property
    def platform_name(self):
        temp_cstr = st_NodeInfo_get_platform_name(self._ptr_node_info)
        if temp_cstr != st_NullChar:
            platform_name = bytes(temp_cstr).decode('utf-8')
        else:
            platform_name = ""
        return platform_name

    @property
    def has_device_name(self):
        return st_NodeInfo_has_device_name(self._ptr_node_info)

    @property
    def device_name(self):
        temp_cstr = st_NodeInfo_get_device_name(self._ptr_node_info)
        if temp_cstr != st_NullChar:
            device_name = bytes(temp_cstr).decode('utf-8')
        else:
            device_name = ""
        return device_name

    @property
    def has_description(self):
        return st_NodeInfo_has_description(self._ptr_node_info)

    @property
    def description(self):
        temp_cstr = st_NodeInfo_get_description(self._ptr_node_info)
        if temp_cstr != st_NullChar:
            description = bytes(temp_cstr).decode('utf-8')
        else:
            description = ""
        return description

    @property
    def is_selected(self):
        return st_NodeInfo_is_selected_node(self._ptr_node_info)

    @property
    def is_default(self):
        return st_NodeInfo_is_default_node(self._ptr_node_info)

    def required_out_string_capacity(self):
        requ_str_capacity = st_NodeInfo_get_required_output_str_length(
            self._ptr_node_info) + 1
        return requ_str_capacity

    def __str__(self):
        return NodeInfoBase.to_string(self.pointer)


class ArgumentBase(object):
    def __init__(self, ptr_argument=st_NullArgumentBase, owns_ptr=True):
        self._ptr_argument = st_NullArgumentBase
        self._owns_ptr = True
        self._last_status = st_ARCH_STATUS_SUCCESS.value

        if ptr_argument != st_NullArgumentBase:
            self._ptr_argument = ptr_argument
            self._owns_ptr = owns_ptr

    def __del__(self):
        if self._ptr_argument != st_NullArgumentBase and self._owns_ptr:
            st_Argument_delete(self._ptr_argument)
            self._owns_ptr = False
        self._ptr_argument = st_NullArgumentBase

    @property
    def pointer(self):
        return self._ptr_argument

    @property
    def arch_id(self):
        return st_Argument_get_arch_id(self._ptr_argument)

    @property
    def arch_str(self):
        arch_str = None
        if st_Argument_has_arch_string(self._ptr_argument):
            arch_str = bytes(st_Argument_get_arch_string(
                self._ptr_argument)).decode('utf-8')
        return arch_str

    @property
    def uses_buffer(self):
        return st_Argument_uses_cobjects_buffer(self._ptr_argument)

    @property
    def ptr_buffer(self):
        ptr_buffer = st_NullBuffer
        if st_Argument_uses_cobjects_buffer(self._ptr_argument):
            ptr_buffer = st_Argument_get_cobjects_buffer(self._ptr_argument)
        return ptr_buffer

    @property
    def buffer_slot_size(self):
        return st_Argument_get_cobjects_buffer_slot_size(self._ptr_argument)

    @property
    def uses_raw_argument(self):
        return st_Argument_uses_raw_argument(self._ptr_argument)

    @property
    def ptr_raw_argument(self):
        return st_Argument_get_ptr_raw_argument(self._ptr_argument)

    @property
    def size(self):
        return st_Argument_get_size(self._ptr_argument)

    @property
    def capacity(self):
        return st_Argument_get_capacity(self._ptr_argument)

    @property
    def has_argument_buffer(self):
        return st_Argument_has_argument_buffer(self._ptr_argument)

    @property
    def requires_argument_buffer(self):
        return st_Argument_requires_argument_buffer(self._ptr_argument)

    @property
    def controller(self):
        return ControllerBase(
            ptr_controller=st_Argument_get_ptr_base_controller(
                self._ptr_argument), owns_ptr=False)

    @property
    def last_status(self):
        return self._last_status

    @property
    def last_status_success(self):
        return self._last_status == st_ARCH_STATUS_SUCCESS.value

    def send(self, buffer=st_NullBuffer, remap_buffer=True,
             ptr_raw_arg_begin=st_Null, raw_arg_size=None):
        ptr_buffer = st_NullBuffer
        if buffer is not None and buffer is not st_NullBuffer:
            if isinstance(buffer, Buffer):
                ptr_buffer = buffer.pointer
            else:
                ptr_buffer = buffer

        if ptr_buffer != st_NullBuffer:
            if remap_buffer:
                self._last_status = st_Argument_send_buffer(
                    self._ptr_argument, ptr_buffer)
            else:
                self._last_status = st_Argument_send_buffer_without_remap(
                    self._ptr_argument, ptr_buffer)
        elif raw_arg_size is not None:
            if ptr_raw_arg_begin == st_Null:
                ptr_raw_arg_begin = self.ptr_raw_argument
            self._last_status = st_Argument_send_raw_argument(
                self._ptr_argument, ptr_raw_arg_begin, raw_arg_size)
        else:
            self._last_status = st_Argument_send_again(self._ptr_argument)

        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful send op; status:{0}".format(
                self._last_status))

        return self

    def send_buffer(self, buffer):
        return self.send(buffer=buffer, remap_buffer=True)

    def send_buffer_without_remap(self, buffer):
        return self.send(buffer=buffer, remap_buffer=False)

    def send_raw_argument(self, ptr_raw_arg_begin, raw_arg_size):
        return self.send(ptr_raw_arg_begin=ptr_raw_arg_begin,
                         raw_arg_size=raw_arg_size)

    def receive(self, buffer=st_NullBuffer, remap_buffer=True,
                ptr_raw_arg_begin=st_Null, raw_arg_capacity=None):
        ptr_buffer = st_NullBuffer
        if buffer is not None and buffer is not st_NullBuffer:
            if isinstance(buffer, Buffer):
                ptr_buffer = buffer.pointer
            elif isinstance(buffer, CBuffer):
                _buffer = Buffer(cbuffer=buffer)
                ptr_buffer = _buffer.pointer
            else:
                ptr_buffer = buffer

        if ptr_buffer != st_NullBuffer:
            if remap_buffer:
                self._last_status = st_Argument_receive_buffer(
                    self._ptr_argument, ptr_buffer)
            else:
                self._last_status = st_Argument_receive_buffer_without_remap(
                    self._ptr_argument, ptr_buffer)
        elif raw_arg_capacity is not None:
            if ptr_raw_arg_begin is st_Null:
                ptr_raw_arg_begin = st_Argument_get_ptr_raw_argument(
                    self._ptr_argument)
            self._last_status = st_Argument_receive_raw_argument(
                self._ptr_argument, ptr_raw_arg_begin, raw_arg_capacity)
        else:
            self._last_status = st_Argument_receive_again(
                self._ptr_argument)

        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful receive op; status:{0}".format(
                self._last_status))

        return self

    def receive_buffer(self, buffer):
        return self.receive(buffer=buffer, remap_buffer=True)

    def receive_buffer_without_remap(self, buffer):
        return self.receive(buffer=buffer, remap_buffer=False)

    def receive_raw_argument(self, ptr_raw_arg_begin, raw_arg_capacity):
        return self.receive(ptr_raw_arg_begin=ptr_raw_arg_begin,
                            raw_arg_capacity=raw_arg_capacity)


class ControllerBase(object):
    def __init__(self, ptr_controller=st_NullControllerBase, owns_ptr=True):
        if ptr_controller != st_NullControllerBase:
            self._ptr_ctrl = ptr_controller
            self._owns_ptr = owns_ptr
        else:
            self._ptr_ctrl = st_NullControllerBase
            self._owns_ptr = True
        self._last_status = st_ARCH_STATUS_SUCCESS.value

    def __del__(self):
        if self._ptr_ctrl != st_NullControllerBase and self._owns_ptr:
            st_Controller_delete(self._ptr_ctrl)
            self._owns_ptr = False
        self._ptr_ctrl = st_NullControllerBase

    @property
    def pointer(self):
        return self._ptr_ctrl

    @property
    def last_status(self):
        return self._last_status

    @property
    def last_status_success(self):
        return self._last_status == st_ARCH_STATUS_SUCCESS.value

    @property
    def uses_nodes(self):
        return st_Controller_uses_nodes(self._ptr_ctrl)

    @property
    def arch_id(self):
        return st_Controller_get_arch_id(self._ptr_ctrl)

    @property
    def arch_str(self):
        arch_str = None
        if st_Controller_has_arch_string(self._ptr_ctrl):
            arch_str = bytes(st_Controller_get_arch_string(
                self._ptr_ctrl)).decode('utf-8')
        return arch_str

    @property
    def config_str(self):
        conf_str = None
        if st_Controller_has_config_string(self._ptr_ctrl):
            conf_str = bytes(st_Controller_get_config_string(
                self._ptr_ctrl)).decode('utf-8')
        return conf_str

    @property
    def ready_to_send(self):
        return st_Controller_is_ready_to_send(self._ptr_ctrl)

    @property
    def ready_to_receive(self):
        return st_Controller_is_ready_to_receive(self._ptr_ctrl)

    @property
    def ready_to_run_kernel(self):
        return st_Controller_is_ready_to_run_kernel(self._ptr_ctrl)

    @property
    def ready_to_remap(self):
        return st_Controller_is_ready_to_remap(self._ptr_ctrl)

    @property
    def has_remap_kernel(self):
        if not st_Controller_is_in_debug_mode(self._ptr_ctrl):
            return st_Controller_has_remap_cobject_buffer_kernel(
                self._ptr_ctrl)
        else:
            return st_Controller_has_remap_cobject_buffer_debug_kernel(
                self._ptr_ctrl)

    @property
    def remap_kernel_id(self):
        if not st_Controller_is_in_debug_mode(self._ptr_ctrl):
            return st_Controller_get_remap_cobject_buffer_kernel_id(
                self._ptr_ctrl)
        else:
            return st_Controller_get_remap_cobject_buffer_debug_kernel_id(
                self._ptr_ctrl)

    @property
    def in_debug_mode(self):
        return st_Controller_is_in_debug_mode(self._ptr_ctrl)

    @property
    def num_kernels(self):
        return st_Controller_get_num_of_kernels(self._ptr_ctrl)

    def enable_debug_mode(self):
        self._last_status = st_Controller_enable_debug_mode(self._ptr_ctrl)
        return self

    def disable_debug_mode(self):
        self._last_status = st_Controller_disable_debug_mode(self._ptr_ctrl)
        return self

    def has_kernel(self, kernel_id):
        return st_Controller_has_kernel_id(
            self._ptr_ctrl, st_kernel_id_t(kernel_id))

    def has_kernel_by_name(self, kernel_name_str):
        kernel_name_str = kernel_name_str.encode('utf-8')
        return st_Controller_has_kernel_by_name(
            self._ptr_ctrl, ct.c_char_p(kernel_name_str))

    def num_kernel_arguments(self, kernel_id):
        return st_Controller_get_num_of_kernel_arguments(
            self._ptr_ctrl, st_kernel_id_t(kernel_id))

    def kernel_name(self, kernel_id):
        _kernel_name = None
        _kernel_name_cstr = st_NullChar
        _kernel_id = st_kernel_id_t(kernel_id)
        if st_Controller_kernel_has_name(self._ptr_ctrl, _kernel_id):
            _kernel_name_cstr = st_Controller_get_kernel_name_string(
                self._ptr_ctrl, _kernel_id)
        if _kernel_name_cstr != st_NullChar:
            _kernel_name = bytes(_kernel_name_cstr.value).decode('utf-8')
        return _kernel_name

    # TODO: Implement missing KernelConfig* methods as soon as all API details
    #       concerning KernelCOnfig are settled!!

    def send(self, ptr_argument, ptr_buffer=st_NullBuffer,
             ptr_raw_arg_begin=st_Null, raw_arg_size=None):
        if ptr_buffer != st_NullBuffer:
            self._last_status = st_Controller_send_buffer(
                self._ptr_ctrl, ptr_argument, ptr_buffer)
        elif ptr_raw_arg_begin != st_Null and raw_arg_size is not None:
            self._last_status = st_Controller_send_detailed(
                self._ptr_ctrl, ptr_argument, ptr_raw_arg_begin,
                st_arch_size_t(raw_arg_size))
        else:
            self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value

        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful send op; status:{0}".format(
                self._last_status))

        return self

    def send_buffer(self, ptr_argument, ptr_buffer):
        return self.send(ptr_argument, ptr_buffer=ptr_buffer)

    def send_detailed(self, ptr_argument, ptr_raw_arg_begin, raw_arg_size):
        return self.send(ptr_argument,
                         ptr_raw_arg_begin, raw_arg_size=raw_arg_size)

    def receive(self, ptr_argument=st_NullArgumentBase,
                ptr_buffer=st_NullBuffer,
                ptr_raw_arg_begin=st_Null, raw_arg_capacity=None):
        self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value
        if ptr_argument != st_NullArgumentBase:
            prev_last_status = self._last_status
            if ptr_buffer != st_NullBuffer:
                self._last_status = st_Controller_receive_buffer(
                    self._ptr_ctrl, ptr_buffer, ptr_argument)
            elif ptr_raw_arg_begin != st_Null and raw_arg_capacity is not None:
                self._last_status = st_Controller_send_detailed(
                    self._ptr_ctrl, ptr_raw_arg_begin, st_arch_size_t(raw_arg_capacity), ptr_argument)

            raise_error_if_status_not_success(
                self._last_status, msg="unsuccessful receive op; status:{0}".format(
                    self._last_status), cls=RuntimeError, prev=prev_last_status)

        return self

    def receive_buffer(self, ptr_buffer, ptr_argument):
        return self.receive(ptr_argument=ptr_argument, ptr_buffer=ptr_buffer)

    def receive_detailed(self, ptr_raw_arg_begin, raw_arg_capacity,
                         ptr_argument):
        return self.receive(
            ptr_argument=ptr_argument,
            ptr_raw_arg_begin=ptr_raw_arg_begin,
            raw_arg_capacity=raw_arg_capacity)

    def buffer_arg_needs_remapping(self, argument):
        needs_remapping = False
        ptr_argument = st_NullArgumentBase
        if argument is not None and isinstance(argument, ArgumentBase):
            ptr_argument = argument.pointer
        else:
            ptr_argument = argument
        if ptr_argument is not None and ptr_argument != st_NullArgumentBase and \
                st_Argument_uses_cobjects_buffer(ptr_argument):
            needs_remapping = not st_Controller_is_cobjects_buffer_arg_remapped(
                self._ptr_ctrl, ptr_argument)
        else:
            raise ValueError("illegal argument supplied")
        return needs_remapping

    def remap_buffer_arg(self, argument):
        ptr_argument = st_NullArgumentBase
        if argument is not None and isinstance(argument, ArgumentBase):
            ptr_argument = argument.pointer
        else:
            ptr_argument = argument

        if ptr_argument is not None and ptr_argument != st_NullArgumentBase:
            self._last_status = st_Controller_remap_cobjects_buffer_arg(
                self._ptr_ctrl, ptr_argument)

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccesful mapping op; status:{0}".format(
                    self._last_status))
        else:
            raise RuntimeError("illegal argument supplied")

        return self


class NodeControllerBase(ControllerBase):
    def __init__(self, ptr_controller=st_NullControllerBase, owns_ptr=True):
        super().__init__(ptr_controller=ptr_controller, owns_ptr=owns_ptr)

    def __del__(self):
        super().__del__()

    @staticmethod
    def num_available_nodes(ctrl=None):
        _ptr_ctrl = st_NullControllerBase
        if ctrl is not None and isinstance(ctrl, NodeControllerBase):
            _ptr_ctrl = ctrl.pointer
        elif ctrl is not None:
            _ptr_ctrl = ctrl

        return _ptr_ctrl != st_NullControllerBase \
            and st_Controller_get_num_available_nodes(_ptr_ctrl) or 0

    @staticmethod
    def available_node_indices(ctrl=None):
        _ptr_ctrl = st_NullControllerBase
        if ctrl is not None and isinstance(ctrl, NodeControllerBase):
            _ptr_ctrl = ctrl.pointer
        elif ctrl is not None:
            _ptr_ctrl = ctrl

        _node_indices = set()
        if _ptr_ctrl != st_NullControllerBase:
            av_num_nodes = NodeControllerBase.num_available_nodes(_ptr_ctrl)

            if av_num_nodes > 0:
                _temp_node_indices_t = st_node_index_t * int(av_num_nodes)
                _temp_node_indices = _temp_node_indices_t()
                _status = st_Controller_get_available_node_indices(
                    _ptr_ctrl, st_arch_size_t(av_num_nodes), _temp_node_indices)
                if _status == st_ARCH_STATUS_SUCCESS.value:
                    for idx in _temp_node_indices:
                        if st_Controller_is_node_available_by_index(
                                _ptr_ctrl, idx):
                            _node_indices.add(idx)

        return list(_node_indices)

    @staticmethod
    def available_node_ids(ctrl=None):
        _node_ids = []
        available_node_indices = \
            NodeControllerBase.available_node_indices(ctrl)

        if available_node_indices and len(available_node_indices) > 0:
            _ptr_ctrl = st_NullControllerBase
            if ctrl is not None and isinstance(ctrl, NodeControllerBase):
                _ptr_ctrl = ctrl.pointer
            elif ctrl is not None:
                _ptr_ctrl = ctrl

            if _ptr_ctrl != st_NullControllerBase:
                for idx in available_node_indices:
                    _ptr_node = st_Controller_get_ptr_node_id_by_index(
                        _ptr_ctrl, st_node_index_t(idx))
                    if _ptr_node != st_NullNodeId:
                        _node_ids.append(NodeId(ext_ptr_node_id=_ptr_node))

        return _node_ids

    @staticmethod
    def available_node_id_strs(ctrl=None):
        _node_id_strs = []
        available_node_indices = \
            NodeControllerBase.available_node_indices(ctrl)

        if available_node_indices and len(available_node_indices) > 0:
            _ptr_ctrl = st_NullControllerBase
            if ctrl is not None and isinstance(ctrl, NodeControllerBase):
                _ptr_ctrl = ctrl.pointer
            elif ctrl is not None:
                _ptr_ctrl = ctrl

            if _ptr_ctrl != st_NullControllerBase:
                for idx in available_node_indices:
                    _ptr_node = st_Controller_get_ptr_node_id_by_index(
                        _ptr_ctrl, st_node_index_t(idx))
                    if _ptr_node != st_NullNodeId:
                        _node_id_strs.append(NodeId.to_string(_ptr_node))

        return _node_id_strs

    @staticmethod
    def available_node_infos(ctrl=None):
        _node_infos = []
        available_node_indices = \
            NodeControllerBase.available_node_indices(ctrl)

        if available_node_indices and len(available_node_indices) > 0:
            _ptr_ctrl = st_NullControllerBase
            if ctrl is not None and isinstance(ctrl, NodeControllerBase):
                _ptr_ctrl = ctrl.pointer
            elif ctrl is not None:
                _ptr_ctrl = ctrl

            if _ptr_ctrl != st_NullControllerBase:
                for idx in available_node_indices:
                    _ptr_info = st_Controller_get_ptr_node_info_base_by_index(
                        _ptr_ctrl, st_node_index_t(idx))
                    if _ptr_info != st_NullNodeInfoBase:
                        _node_infos.append(NodeInfoBase(
                            ptr_node_info=_ptr_info, owns_ptr=False))

        return _node_infos

    @property
    def num_nodes(self):
        return NodeControllerBase.num_available_nodes(self.pointer)

    @property
    def min_node_index(self):
        return st_Controller_get_min_available_node_index(self._ptr_ctrl)

    @property
    def max_node_index(self):
        return st_Controller_get_max_available_node_index(self._ptr_ctrl)

    @property
    def has_default_node(self):
        return st_Controller_has_default_node(self._ptr_ctrl)

    @property
    def default_node_id(self):
        return NodeId(ext_ptr_node_id=st_Controller_get_default_node_id(
            self._ptr_ctrl))

    @property
    def default_node_id_str(self):
        return NodeId.to_string(
            st_Controller_get_default_node_id(self._ptr_ctrl))

    @property
    def default_node_index(self):
        return st_Controller_get_default_node_index(self._ptr_ctrl)

    @property
    def default_node_info(self):
        return NodeInfoBase(
            ptr_node_info=st_Controller_get_default_node_info_base(
                self._ptr_ctrl),
            owns_ptr=False)

    @property
    def has_selected_node(self):
        return st_Controller_has_selected_node(self._ptr_ctrl)

    @property
    def selected_node_id(self):
        return NodeId(ext_ptr_node_id=st_Controller_get_ptr_selected_node_id(
            self._ptr_ctrl))

    @property
    def selected_node_id_str(self):
        return NodeId.to_string(st_Controller_get_ptr_selected_node_id(
            self._ptr_ctrl))

    @property
    def selected_node_index(self):
        return st_Controller_get_selected_node_index(self._ptr_ctrl)

    @property
    def selected_node_info(self):
        return NodeInfoBase(
            ptr_node_info=st_Controller_get_ptr_selected_node_info_base(
                self._ptr_ctrl), owns_ptr=False)

    @property
    def can_change_node(self):
        return st_Controller_can_change_selected_node(self._ptr_ctrl)

    @property
    def can_change_node_directly(self):
        return st_Controller_can_directly_change_selected_node(self._ptr_cltr)

    @property
    def can_unselect_node(self):
        return st_Controller_can_unselect_node(self._ptr_ctrl)

    # -------------------------------------------------------------------------

    def is_default_node_check(
            self,
            node_id=st_NullNodeId,
            node_id_str=None,
            node_index=None,
            platform_id=None,
            device_id=None):
        _is_default_node = False
        if node_id is not None and isinstance(node_id, NodeId):
            _is_default_node = st_Controller_is_default_node_id(
                self._ptr_ctrl, node_id.pointer)
        elif node_id is not None and node_id != st_NullNodeId:
            _is_default_node = st_Controller_is_default_node_id(
                self._ptr_ctrl, node_id)
        elif node_id_str is not None:
            node_id_str = node_id_str.encode('utf-8')
            _is_default_node = st_Controller_is_default_node(
                self._ptr_ctrl, ct.c_char_p(node_id_str))
        elif node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
            _is_default_node = st_Controller_is_default_node_index(
                self._ptr_ctrl, st_node_index_t(node_index))
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            _is_default_node = \
                st_Controller_is_default_platform_id_and_device_id(
                    self._ptr_ctrl, st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))
        return _is_default_node

    def is_default_node(self, node_id_str):
        return self.is_default_node_check(node_id_str=node_id_str)

    def is_default_node_index(self, node_index):
        return self.is_default_node_check(node_index=node_index)

    def is_default_node_id(self, node_id):
        return self.is_default_node_check(node_id=node_id)

    def are_default_node_by_platform_id_and_device_id(self, pid, dev_id):
        return self.is_default_node_check(platform_id=pid, device_id=dev_id)

    # -------------------------------------------------------------------------

    def get_available_node_indices(self):
        return NodeControllerBase.available_node_indices(self.pointer)

    def get_available_node_ids(self):
        return NodeControllerBase.available_node_ids(self.pointer)

    def get_available_node_id_strs(self):
        return NodeControllerBase.available_node_id_strs(self.pointer)

    def get_available_node_infos(self):
        return NodeControllerBase.available_node_infos(self.pointer)

    def __str__(self):
        return '\r\n'.join(
            [str(info) for info in self.get_available_node_infos()])

    # -------------------------------------------------------------------------

    def get_node_info_disp(self, node_id=st_NullNodeId, node_id_str=None,
                           node_index=None, platform_id=None, device_id=None):
        node_info_base = None
        _ptr_node_info = st_NullNodeInfoBase
        if node_id_str is not None:
            node_id_str = node_id_str.encode('utf-8')
            _ptr_node_info = st_Controller_get_ptr_node_info_base(
                self._ptr_ctrl, ct.c_char_p(node_id_str))
        elif node_id is not None and isinstance(node_id, NodeId):
            _ptr_node_info = st_Controller_get_ptr_node_info_base_by_node_id(
                self._ptr_ctrl, node_id.pointer)
        elif node_id is not None and node_id != st_NullNodeId:
            _ptr_node_info = st_Controller_get_ptr_node_info_base_by_node_id(
                self._ptr_ctrl, node_id)
        elif node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
            _ptr_node_info = st_Controller_get_ptr_node_info_base_by_index(
                self._ptr_ctrl, st_node_index_t(node_index))
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            _ptr_node_info = \
                st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id(
                    self._ptr_ctrl, st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))

        if _ptr_node_info != st_NullNodeInfoBase:
            node_info_base = NodeInfoBase(
                ptr_node_info=_ptr_node_info, owns_ptr=False)
        return node_info_base

    def get_node_info(self, node_id_str):
        return self.get_node_info_disp(node_id_str=node_id_str)

    def get_node_info_by_id(self, node_id):
        return self.get_node_info_disp(node_id=node_id)

    def get_node_info_by_index(self, node_index):
        return self.get_node_info_disp(node_index=node_index)

    def get_node_info_by_platform_id_and_device_id(self, pid, dev_id):
        return self.get_node_info_disp(platform_id=pid, device_id=dev_id)

    # -------------------------------------------------------------------------

    def is_selected_node_check(
            self,
            node_id_str=None,
            node_id=st_NullNodeId,
            node_index=None,
            platform_id=None,
            device_id=None):
        _is_selected_node = False
        _selected_node_index = st_Controller_get_selected_node_index(
            self._ptr_ctrl)

        if st_Controller_has_selected_node(self._ptr_ctrl) and \
           _selected_node_index != st_NODE_UNDEFINED_INDEX.value:
            if node_index is not None and \
                    node_index != st_NODE_UNDEFINED_INDEX.value:
                is_selected_node = bool(node_index == _selected_node_index)
            elif node_id is not None and isinstance(node_id, NodeId):
                _is_selected_node = bool(
                    _selected_node_index == st_Controller_get_node_index_by_node_id(
                        self._ptr_ctrl, node_id.pointer))
            elif node_id is not None and node_id != st_NullNodeId:
                _is_selected_node = bool(
                    _selected_node_index == st_Controller_get_node_index_by_node_id(
                        self._ptr_ctrl, node_id))
            elif node_id_str is not None:
                node_id_str = node_id_str.encode('utf-8')
                _is_selected_node = bool(
                    _selected_node_index == st_Controller_get_node_index(
                        self._ptr_ctrl, ct.c_char_p(node_id_str)))
            elif platform_id is not None and \
                    platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                    device_id is not None and \
                    device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
                _is_selected_node = bool(
                    _selected_node_index == st_Controller_get_node_index_by_platform_id_and_device_id(
                        self._ptr_ctrl,
                        st_node_platform_id_t(platform_id),
                        st_node_device_id_t(device_id)))

        return _is_selected_node

    def is_selected_node(self, node_id_str):
        return self.is_selected_node_check(node_id_str=node_id_str)

    def is_selected_node_index(self, node_index):
        return self.is_selected_node_check(node_index=node_index)

    def is_selected_node_id(self, node_id):
        return self.is_selected_node_check(node_id=node_id)

    def are_selected_node_platform_id_and_device_id(self, pid, dev_id):
        return self.is_selected_node_check(platform_id=pid, device_id=dev_id)

    # -------------------------------------------------------------------------

    def is_node_available_check(
            self,
            node_id_str=None,
            node_id=st_NullNodeId,
            node_index=None,
            platform_id=None,
            device_id=None):
        _is_available = False
        if node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
            _is_available = st_Controller_is_node_available_by_index(
                self._ptr_ctrl, st_node_index_t(node_index))
        elif node_id is not None and isinstance(node_id, NodeId):
            _is_available = st_Controller_is_node_available_by_node_id(
                self._ptr_ctrl, node_id.pointer)
        elif node_id is not None and node_id != st_NullNodeId:
            _is_available = st_Controller_is_node_available_by_node_id(
                self._ptr_ctrl, node_id)
        elif node_id_str is not None:
            node_id_str = node_id_str.encode('utf-8')
            _is_available = st_Controller_is_node_available(
                self._ptr_ctrl, ct.c_char_p(node_id_str))
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            _is_available = \
                st_Controller_is_node_available_by_platform_id_and_device_id(
                    self._ptr_ctrl, st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))

        return _is_available

    def is_node_available(self, node_id_str):
        return self.is_node_available_check(node_id_str=node_id_str)

    def is_node_index_available(self, node_index):
        return self.is_node_available_check(node_index=node_index)

    def is_node_id_available(self, node_id):
        return self.is_node_available_check(node_id=node_id)

    def are_platform_id_and_device_id_available(self, pid, dev_id):
        return self.is_node_available_check(platform_id=pid, device_id=dev_id)

    # ------------------------------------------------------------------------

    def select_node_disp(self, node_id_str=None, node_id=st_NullNodeId,
                         node_index=None, platform_id=None, device_id=None):
        prev_last_status = self._last_status
        if node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
            self._last_status = st_Controller_select_node_by_index(
                self._ptr_ctrl, st_node_index_t(node_index))
        elif node_id is not None and isinstance(node_id, NodeId):
            self._last_status = st_Controller_select_node_by_node_id(
                self._ptr_ctrl, node_id.pointer)
        elif node_id is not None and node_id != st_NullNodeId:
            self._last_status = st_Controller_select_node_by_node_id(
                self._ptr_ctrl, node_id)
        elif node_id_str is not None:
            node_id_str = node_id_str.encode('utf-8')
            self._last_status = st_Controller_select_node(
                self._ptr_ctrl, ct.c_char_p(node_id_str))
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            self._last_status = \
                st_Controller_select_node_by_plaform_id_and_device_id(
                    self._ptr_ctrl, st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))

        raise_error_if_status_not_success(
            self._last_status,
            msg="unsuccessful select node op; status:{0}".format(
                self._last_status),
            cls=RuntimeError,
            prev=prev_last_status)

        return self

    def select_node(self, node_id_str):
        return self.select_node_disp(node_id_str=node_id_str)

    def select_node_by_index(self, node_index):
        return self.select_node_disp(node_index=node_index)

    def select_node_by_id(self, node_id):
        return self.select_node_disp(node_id=node_id)

    def select_node_by_platform_id_and_device_id(self, pid, dev_id):
        return self.select_node_disp(platform_id=pid, device_id=dev_id)

    # ------------------------------------------------------------------------

    def change_selected_node(self, current_node_index, new_selected_index):
        self._last_status = st_Controller_change_selected_node(
            self._ptr_ctrl, st_node_index_t(current_node_index),
            st_node_index_t(new_selected_index))

        raise_error_if_status_not_success(
            self._last_status,
            "unsuccessful change selected node op; status:{0}".format(
                self._last_status))

        return self

    # ------------------------------------------------------------------------

    def unselect_node_disp(self, node_id_str=None, node_id=st_NullNodeId,
                           node_index=None, platform_id=None, device_id=None):
        prev_last_status = self._last_status
        if node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
            self._last_status = st_Controller_unselect_node_by_index(
                self._ptr_ctrl, st_node_index_t(node_index))
        elif node_id is not None and isinstance(node_id, NodeId):
            self._last_status = st_Controller_unselect_node_by_node_id(
                self._ptr_ctrl, node_id.pointer)
        elif node_id is not None and node_id != st_NullNodeId:
            self._last_status = st_Controller_unselect_node_by_node_id(
                self._ptr_ctrl, node_id)
        elif node_id_str is not None:
            node_id_str = node_id_str.encode('utf-8')
            self._last_status = st_Controller_unselect_node(
                self._ptr_ctrl, ct.c_char_p(node_id_str))
        elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID.value:
            self._last_status = \
                st_Controller_unselect_node_by_platform_id_and_device_id(
                    self._ptr_ctrl, st_node_platform_id_t(platform_id),
                    st_node_device_id_t(device_id))
        raise_error_if_status_not_success(
            self._last_status,
            msg="unsuccessful unselect node op; status:{0}".format(
                self._last_status),
            cls=RuntimeError,
            prev=prev_last_status)

        return self

    def unselect_node(self, node_id_str):
        return self.unselect_node_disp(node_id_str=node_id_str)

    def unselect_node_by_index(self, node_index):
        return self.unselect_node_disp(node_index=node_index)

    def unselect_node_by_id(self, node_id):
        return self.unselect_node_disp(node_id=node_id)

    def unselect_node_by_platform_id_and_device_id(self, pid, dev_id):
        return self.unselect_node_disp(platform_id=pid, device_id=dev_id)
