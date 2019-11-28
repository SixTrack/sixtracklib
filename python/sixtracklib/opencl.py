#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer, CObject
from .particles import ParticlesSet
from .buffer import Buffer
from .config import SIXTRACKLIB_MODULES
from .control import raise_error_if_status_not_success

if SIXTRACKLIB_MODULES.get('opencl', False):
    from .stcommon import st_ClContextBase_p, st_NullClContextBase, \
        st_NullBuffer, st_Buffer_p, st_Null, st_NullChar, \
        st_ARCH_ILLEGAL_PROGRAM_ID, st_ARCH_ILLEGAL_KERNEL_ID, \
        st_ARCH_STATUS_GENERAL_FAILURE, st_ARCH_STATUS_SUCCESS, \
        st_arch_program_id_t, st_arch_kernel_id_t, st_arch_status_t, \
        st_arch_size_t, st_node_platform_id_t, st_node_device_id_t, \
        st_ClNodeId_p, st_NullClNodeId, st_ClNodeInfo_p, st_NullClNodeInfo, \
        st_ComputeNodeId_get_platform_id, st_ComputeNodeId_get_device_id, \
        st_ComputeNodeId_set_platform_id, st_ComputeNodeId_set_device_id, \
        st_ComputeNodeId_is_valid, st_ComputeNodeId_compare, \
        st_ComputeNodeId_are_equal, st_ComputeNodeId_from_string, \
        st_ComputeNodeInfo_preset, st_ComputeNodeInfo_print_out, \
        st_ComputeNodeInfo_free, st_ComputeNodeInfo_delete, \
        st_ComputeNodeInfo_reserve, st_ComputeNodeInfo_make, \
        st_ComputeNodeInfo_is_valid, st_ComputeNodeInfo_get_id, \
        st_ComputeNodeInfo_get_platform_id, st_ComputeNodeInfo_get_device_id, \
        st_ComputeNodeInfo_get_arch, st_ComputeNodeInfo_get_platform, \
        st_ComputeNodeInfo_get_name, st_ComputeNodeInfo_get_description, \
        st_ClContext_create, st_ClContextBase_delete, \
        st_ClContextBase_has_selected_node, \
        st_ClContextBase_get_selected_node_id, \
        st_ClContextBase_get_selected_node_info, \
        st_ClContextBase_get_selected_node_index, \
        st_ClContextBase_get_selected_node_id_str, \
        st_ClContextBase_select_node_by_node_id, \
        st_ClContextBase_select_node_by_index, st_ClContextBase_select_node, \
        st_ClContextBase_get_num_available_programs, \
        st_ClContextBase_print_nodes_info, \
        st_ClContextBase_add_program_file, st_ClContextBase_compile_program, \
        st_ClContextBase_get_num_available_kernels, \
        st_ClContextBase_enable_kernel, \
        st_ClContextBase_find_kernel_id_by_name, \
        st_ClContextBase_get_kernel_local_mem_size, \
        st_ClContextBase_get_kernel_num_args, \
        st_ClContextBase_get_kernel_work_group_size, \
        st_ClContextBase_get_kernel_max_work_group_size, \
        st_ClContextBase_set_kernel_work_group_size, \
        st_ClContextBase_get_kernel_preferred_work_group_size_multiple, \
        st_ClContextBase_get_ptr_kernel_argument, \
        st_ClContextBase_get_kernel_argument_type, \
        st_ClContextBase_reset_single_kernel_argument, \
        st_ClContextBase_reset_kernel_arguments, \
        st_ClContextBase_assign_kernel_argument, \
        st_ClContextBase_assign_kernel_argument_value, \
        st_ClContextBase_assign_kernel_argument_ptr, \
        st_ClContextBase_calculate_kernel_num_work_items, \
        st_ClContextBase_run_kernel, st_ClContextBase_run_kernel_wgsize, \
        st_ClContextBase_get_kernel_exec_counter, \
        st_ClContextBase_get_last_exec_time, st_ClContextBase_get_min_exec_time, \
        st_ClContextBase_get_max_exec_time, st_ClContextBase_get_avg_exec_time, \
        st_ClContextBase_get_last_exec_work_group_size, \
        st_ClContextBase_get_last_exec_num_work_items, \
        st_ClContextBase_reset_kernel_exec_timing, \
        st_ClContextBase_get_program_id_by_kernel_id, \
        st_ClContextBase_has_remapping_program, \
        st_ClContextBase_remapping_program_id, \
        st_ClContextBase_has_remapping_kernel, \
        st_ClContextBase_remapping_kernel_id, \
        st_ClContextBase_set_remapping_kernel_id, \
        st_ClContextBase_is_debug_mode_enabled, \
        st_ClContextBase_enable_debug_mode, \
        st_ClContextBase_disable_debug_mode, \
        st_ClArgument_p, st_NullClArgument, st_ClArgument_new_from_buffer, \
        st_ClArgument_delete, st_ClArgument_write, st_ClArgument_read, \
        st_ClArgument_write_memory, st_ClArgument_read_memory, \
        st_ClArgument_get_argument_size, st_ClArgument_uses_cobj_buffer, \
        st_ClArgument_get_ptr_cobj_buffer, st_ClArgument_get_ptr_to_context, \
        st_ClArgument_attach_to_context

    class ClController(object):
        def __init__(self, config_str=None, device_id=None,
                     ext_ptr_ctrl=st_NullClContextBase, owns_ptr=True):
            self._ptr_ctrl = st_NullClContextBase
            self._owns_ctrl = True
            self._last_status = st_ARCH_STATUS_SUCCESS.value

            if ext_ptr_ctrl != st_NullClContextBase:
                self._ptr_ctrl = ext_ptr_ctrl
                self._owns_ctrl = owns_ptr
            else:
                self._ptr_ctrl = st_ClContext_create()

            if self._ptr_ctrl is st_NullClContextBase:
                raise RuntimeError("ClController needs a valid ptr")

            if device_id is not None:
                device_id = device_id.strip()
                if device_id.startswith("opencl:"):
                    device_id = device_id[7:]
                elif ":" in device_id:
                    device_id = None

            if device_id is not None and len(device_id) > 0:
                device_id.encode('utf-8')
                st_ClContextBase_select_node(
                    self._ptr_ctrl, ct.c_char_p(device_id))

        def __del__(self):
            if self._owns_ctrl and self._ptr_ctrl != st_NullClContextBase:
                st_ClContextBase_delete(self._ptr_ctrl)
            self._ptr_ctrl = st_NullClContextBase

        @property
        def last_status(self):
            return self.last_status

        @property
        def pointer(self):
            return self._ptr_ctrl

        @property
        def owns_controller(self):
            return self._owns_ctrl

        @property
        def has_selected_node(self):
            return st_ClContextBase_has_selected_node(self._ptr_ctrl)

        @property
        def selected_node_platform_id(self):
            platform_id = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                platform_id = st_ComputeNodeInfo_get_platform_id(_info)
            return platform_id

        @property
        def selected_node_device_id(self):
            device_id = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                device_id = st_ComputeNodeInfo_get_device_id(_info)
            return device_id

        @property
        def selected_node_id_str(self):
            node_id_str = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                platform_id = st_ComputeNodeInfo_get_platform_id(_info)
                device_id = st_ComputeNodeInfo_get_device_id(_info)
                node_id_str = f"opencl:{platform_id}.{device_id}"
            return node_id_str

        @property
        def selected_node_platform(self):
            node_platform_str = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                _platform_c_str = st_ComputeNodeInfo_get_platform(_info)
                if _platform_c_str != st_NullChar:
                    node_platform_str = bytes(
                        _platform_c_str.value).decode('utf-8')
            return node_platform_str

        @property
        def selected_node_name(self):
            node_name = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                _name_c_str = st_ComputeNodeInfo_get_name(_info)
                if _name_c_str != st_NullChar:
                    node_name = bytes(_name_c_str.value).decode('utf-8')
            return node_name

        @property
        def selected_node_description(self):
            description = None
            _info = st_ClContextBase_get_selected_node_info(self._ptr_ctrl)
            if _info != st_NullClNodeInfo:
                _desc_c_str = st_ComputeNodeInfo_get_name(_info)
                if _desc_c_str != st_NullChar:
                    description = bytes(_desc_c_str.value).decode('utf-8')
            return description

        def add_program_file(self, path_to_program, compile_defs, compile=True):
            program_id = st_ARCH_ILLEGAL_PROGRAM_ID
            if self._ptr_ctrl != st_NullClContextBase:
                path_to_program = path_to_program.strip()
                path_to_program.encode('utf-8')
                compile_defs = compile_defs.strip()
                compile_defs.encode('utf-8')
                program_id = st_ClContextBase_add_program_file(
                    self._ptr_ctrl, ct.c_char_p(path_to_program),
                    ct.c_char_p(compile_defs))
                if compile:
                    if not self.compile_program(program_id):
                        raise RuntimeError("Error while compiling program")
            return program_id

        def compile_program(self, program_id):
            success = False
            if self._ptr_ctrl != st_NullClContextBase and \
                    program_id != st_ARCH_ILLEGAL_PROGRAM_ID.value:
                success = st_ClContextBase_compile_program(
                    self._ptr_ctrl, st_arch_program_id(program_id))
            return success

        def enable_kernel(self, program_id, kernel_name):
            kernel_id = st_ARCH_ILLEGAL_KERNEL_ID.value
            kernel_name = kernel_name.strip()
            kernel_name.encode('utf-8')
            if self._ptr_ctrl != st_NullClContextBase and \
                    program_id != st_ARCH_ILLEGAL_PROGRAM_ID.value:
                kernel_id = st_ClContextBase_enable_kernel(
                    self._ptr_ctrl, ct.c_char_p(kernel_name),
                    st_arch_program_id(program_id))
            return kernel_id

        def find_kernel_by_name(self, kernel_name):
            kernel_id = st_ARCH_ILLEGAL_KERNEL_ID.value
            if self._ptr_ctrl is st_NullClContextBase:
                kernel_name = kernel_name.strip()
                kernel_name.encode('utf-8')
                kernel_id = st_ClContextBase_find_kernel_id_by_name(
                    self._ptr_ctrl, ct.c_char_p(kernel_name))
            return kernel_id

        def set_kernel_arg(self, kernel_id, arg_index, arg):
            if kernel_id != st_ARCH_ILLEGAL_KERNEL_ID.value and \
                    self._ptr_ctrl != st_NullClContextBase:
                if isinstance(arg, ClArgument):
                    st_ClContextBase_assign_kernel_argument(
                        self._ptr_ctrl, st_arch_kernel_id(kernel_id),
                        st_arch_size_t(arg_index), arg.ptr_argument)
                elif isinstance(arg, type(st_ClArgument_p)):
                    st_ClContextBase_assign_kernel_argument(
                        self._ptr_ctrl, st_arch_kernel_id(kernel_id),
                        st_arch_size_t(arg_index), arg)
                else:
                    raise ValueError(
                        "arg expected to be an instance of ClArgument")
            return self

        def set_kernel_arg_value(self, kernel_id, arg_index, val_p, val_size):
            if kernel_id != st_ARCH_ILLEGAL_KERNEL_ID.value and \
                    isinstance(val_p, type(ct.c_void_p)) and val_size > 0:
                st_ClContextBase_assign_kernel_argument_value(
                    self._ptr_ctrl, st_arch_kernel_id(kernel_id),
                    st_arch_size_t(arg_index), val_p, st_arch_size_t(val_size))
            return self

        def reset_kernel_args(self, kernel_id, arg_index=None):
            _kernel_id = st_arch_kernel_id_t(kernel_id)
            status = st_ARCH_STATUS_GENERAL_FAILURE
            if arg_index is not None:
                st_ClContextBase_reset_single_kernel_argument(
                    self._ptr_ctrl, _kernel_id, st_arch_size_t(arg_index))
                status = st_ARCH_STATUS_SUCCESS.value
            else:
                st_ClContextBase_reset_kernel_arguments(
                    self._ptr_ctrl, _kernel_id)
                status = st_ARCH_STATUS_SUCCESS.value
            self._last_status = status
            return self

        def has_kernel(self, kernel_id):
            return kernel_id != st_ARCH_ILLEGAL_KERNEL_ID.value and \
                kernel_id < st_ClContextBase_get_num_available_kernels(
                    self._ptr_ctrl)

        def kernel_local_mem_size(self, kernel_id):
            return st_ClContextBase_get_kernel_local_mem_size(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id))

        def num_kernel_args(self, kernel_id):
            return st_ClContextBase_get_kernel_num_args(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id))

        def kernel_workgroup_size(self, kernel_id):
            return st_ClContextBase_get_kernel_work_group_size(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id))

        def kernel_max_workgroup_size(self, kernel_id):
            return st_ClContextBase_get_kernel_max_work_group_size(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id))

        def kernel_preferred_workgroup_size_multiple(self, kernel_id):
            return st_ClContextBase_get_kernel_preferred_work_group_size_multiple(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id))

        def argument_of_kernel(self, kernel_id, arg_index):
            _ptr_arg = st_ClContextBase_get_ptr_kernel_argument(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id),
                st_arch_size_t(arg_index))
            if _ptr_arg != st_NullClArgument:
                return ClArgument(ext_ptr_arg=_ptr_arg, owns_ptr=False)
            else:
                error_msg = f"""
                "unable to retrieve ClArgument of kernel id={kernel_id},
                argument idx = {arg_index}
                """
                raise ValueError(error_msg)
                self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value

        def argument_type_of_kernel(self, kernel_id, arg_index):
            return st_ClContextBase_get_kernel_argument_type(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id),
                st_arch_size_t(arg_index))

        def set_kernel_workgroup_size(self, kernel_id, wgsize):
            ret = st_ClContextBase_set_kernel_work_group_size(
                self._ptr_ctrl, st_arch_kernel_id_t(kernel_id),
                st_arch_kernel_id_t(wgsize))
            if not ret:
                self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value
                error_msg = f"""
                unable to set workgroup size to {wgsize}
                for kernel id={kernel_id}
                """
                raise RuntimeError(error_msg)
            return self

        def run_kernel(self, kernel_id, num_work_items, work_group_size=None):
            _kernel_id = st_arch_kernel_id_t(kernel_id)
            _num_work_items = st_arch_size_t(num_work_items)
            ret = False
            if work_group_size is None:
                ret = st_ClContextBase_run_kernel(
                    self._ptr_ctrl, _kernel_id, _num_work_items)
            else:
                ret = st_ClContextBase_run_kernel_wgsize(
                    self._ptr_ctrl, _kernel_id, _num_work_items,
                    st_arch_size_t(work_group_size))
            if ret:
                self._last_status = st_ARCH_STATUS_SUCCESS.value
            else:
                self._last_status = st_ARCH_STATUS_GENERAL_FAILURE.value
                error_msg = f"Unable to run OpenCL kernel id={kernel_id}"
                raise RuntimeError(error_msg)
            return self

    # -------------------------------------------------------------------------

    class ClArgument(object):
        def __init__(self, buffer=None, ctrl=None,
                     ext_ptr_arg=st_NullClArgument, owns_ptr=True):
            self._ptr_arg = st_NullClArgument
            self._owns_arg = True
            self._last_status = st_ARCH_STATUS_SUCCESS

        def __del__(self):
            if self._owns_arg and self._ptr_arg != st_NullClArgument:
                st_ClArgument_delete(self._ptr_arg)
            self._ptr_arg = st_NullClArgument

        @property
        def pointer(self):
            return self._ptr_arg

        @property
        def owns_argument(self):
            return self._owns_arg

        @property
        def controller(self):
            _ptr_ctrl = st_ClArgument_get_ptr_to_context(self._ptr_arg)
            return ClController(ext_ptr_ctrl=_ptr_ctrl, owns_ptr=False)

        @property
        def uses_buffer(self):
            return st_ClArgument_uses_cobj_buffer(self._ptr_arg)

        @property
        def ptr_buffer(self):
            _ptr_buffer = st_NullBuffer
            if st_ClArgument_uses_cobj_buffer(self._ptr_arg):
                _ptr_buffer = st_ClArgument_get_ptr_cobj_buffer(self._ptr_arg)
            if _ptr_buffer is st_NullBuffer:
                raise RuntimeError("Unable to retrieve buffer from argument")
            return Buffer(ptr_ext_buffer=_ptr_buffer, owns_ptr=False)

        @property
        def size(self):
            return st_ClArgument_get_argument_size(self._ptr_arg)

        @property
        def capacity(self):
            return st_ClArgument_get_argument_size(self._ptr_arg)

        def send(self, buffer=st_NullBuffer, remap_buffer=True,
                 ptr_raw_arg_begin=st_Null, raw_arg_size=None):
            status = st_ARCH_STATUS_GENERAL_FAILURE.value
            ptr_buffer = st_NullBuffer
            if buffer is not None and buffer is not st_NullBuffer:
                if isinstance(buffer, Buffer):
                    ptr_buffer = buffer.pointer
                else:
                    ptr_buffer = buffer

            if ptr_buffer is not st_NullBuffer:
                if remap_buffer:
                    if st_ClArgument_write(self._ptr_argument, ptr_buffer):
                        status = st_ARCH_STATUS_SUCCESS.value
                else:
                    raise ValueError(
                        "sending with remap_buffer=False not yet implemented")
            elif raw_arg_size is not None:
                raise ValueError(
                    "sending raw-memory based ClArguments not yet implemented")

            self._last_status = status
            raise_error_if_status_not_success(
                self._last_status, "unsuccessful send op; status:{0}".format(
                    status))

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
        status = st_ARCH_STATUS_GENERAL_FAILURE.value
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
                status = st_ClArgument_read(self._ptr_arg, ptr_buffer)
            else:
                raise RuntimeError(
                    "receiving with remap_buffer=False not yet implemented")
        elif raw_arg_capacity is not None:
            raise RuntimeError(
                "receiving raw-memory based ClArguments not yet implemented")

        raise_error_if_status_not_success(
            status, "unsuccessful receive op; status:{0}".format(status))
        self._last_status = status

        return self

    def receive_buffer(self, buffer):
        return self.receive(buffer=buffer, remap_buffer=True)

    def receive_buffer_without_remap(self, buffer):
        return self.receive(buffer=buffer, remap_buffer=False)

    def receive_raw_argument(self, ptr_raw_arg_begin, raw_arg_capacity):
        return self.receive(ptr_raw_arg_begin=ptr_raw_arg_begin,
                            raw_arg_capacity=raw_arg_capacity)
else:

    class ClController(object):
        pass

    class ClArgument(object):
        pass
