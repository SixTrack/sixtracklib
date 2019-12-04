#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer, CObject
from .particles import ParticlesSet
from .buffer import Buffer
from .config import SIXTRACKLIB_MODULES
from .control import NodeId, NodeInfoBase, ArgumentBase, NodeControllerBase
from .control import raise_error_if_status_not_success

if SIXTRACKLIB_MODULES.get('cuda', False):
    from .stcommon import st_Null, st_node_platform_id_t, st_node_device_id_t, \
        st_node_index_t, st_arch_status_t, st_arch_size_t, st_buffer_size_t, \
        st_ctrl_size_t, st_ctrl_status_t, st_NullBuffer, \
        st_NullChar, st_ARCH_STATUS_GENERAL_FAILURE, st_ARCH_STATUS_SUCCESS, \
        st_NODE_UNDEFINED_INDEX, st_NODE_ILLEGAL_PLATFORM_ID, \
        st_NODE_ILLEGAL_DEVICE_ID, st_NullNodeId

    from .stcommon import st_CudaNodeInfo_p, st_NullCudaNodeInfo, \
        st_cuda_dev_index_t, st_CudaNodeInfo_new, st_CudaNodeInfo_new_detailed,\
        st_NodeId_create, st_NodeId_delete, \
        st_CudaNodeInfo_get_cuda_device_index, \
        st_CudaNodeInfo_get_pci_bus_id_str, st_CudaNodeInfo_get_warp_size, \
        st_CudaNodeInfo_get_compute_capability, \
        st_CudaNodeInfo_get_num_multiprocessors, \
        st_CudaNodeInfo_get_max_threads_per_block, \
        st_CudaNodeInfo_get_max_threads_per_multiprocessor

    class CudaNodeInfo(NodeInfoBase):
        def __init__(self, ptr_node_info=st_NullCudaNodeInfo, owns_ptr=False,
                     cuda_dev_index=None, platform_id=None,
                     device_id=None, node_index=None, is_default_node=False,
                     is_selected_node=False, **kwargs):
            _ptr_node_info = ptr_node_info
            if _ptr_node_info == st_NullCudaNodeInfo:
                if cuda_dev_index is not None and cuda_dev_index > 0:
                    if platform_id is None:
                        platform_id = st_NODE_ILLEGAL_PLATFORM_ID.value
                    if device_id is None:
                        device_id = st_NODE_ILLEGAL_DEVICE_ID.value
                    if node_index is None:
                        node_index = st_NODE_UNDEFINED_INDEX.value

                    _ptr_node_info = st_CudaNodeInfo_new_detailed(
                        st_cuda_dev_index_t(cuda_dev_index),
                        st_node_platform_id_t(platform_id),
                        st_node_device_id_t(device_id),
                        st_node_index_t(node_index),
                        ct.c_bool(is_default_node),
                        ct.c_bool(is_selected_node))

                    owns_ptr = bool(_ptr_node_info != st_NullCudaNodeInfo)

            if _ptr_node_info != st_NullCudaNodeInfo:
                super().__init__(
                    ptr_node_info=_ptr_node_info, owns_ptr=owns_ptr, **kwargs)

        def __del__(self):
            super().__del__()

        @property
        def cuda_device_index(self):
            return st_CudaNodeInfo_get_cuda_device_index(self._ptr_node_info)

        @property
        def pci_bus_id(self):
            _pci_bus_id_str = None
            _ptr_bus_id_cstr = st_CudaNodeInfo_get_pci_bus_id_str(
                self._ptr_node_info)
            if _ptr_bus_id_cstr != st_NullChar:
                _pci_bus_id_str = bytes(
                    _ptr_bus_id_cstr.value).decode('utf-8')
            return _pci_bus_id_str

        @property
        def warp_size(self):
            return st_CudaNodeInfo_get_warp_size(self._ptr_node_info)

        @property
        def compute_capability(self):
            return st_CudaNodeInfo_get_compute_capability(
                self._ptr_node_info)

        @property
        def num_multiprocessors(self):
            return st_CudaNodeInfo_get_num_multiprocessors(
                self._ptr_node_info)

        @property
        def max_threads_per_block(self):
            return st_CudaNodeInfo_get_max_threads_per_block(
                self._ptr_node_info)

        @property
        def max_threads_per_multiprocessor(self):
            st_CudaNodeInfo_get_max_threads_per_multiprocessor(
                self._ptr_node_info)

    from .stcommon import \
        st_node_id_str_fmt_t, \
        st_CudaController_p, st_NullCudaController, \
        st_ControllerBase_p, st_NullControllerBase, \
        st_CudaController_create, st_CudaController_new, \
        st_CudaController_new_from_node_id, \
        st_CudaController_new_from_platform_id_and_device_id, \
        st_CudaController_new_from_node_index, \
        st_CudaController_new_from_cuda_device_index, \
        st_CudaController_select_node_by_cuda_device_index, \
        st_CudaController_select_node_by_cuda_pci_bus_id, \
        st_CudaController_is_managed_cobject_buffer_remapped, \
        st_CudaController_remap_managed_cobject_buffer, \
        st_CudaController_send_memory, st_CudaController_receive_memory, \
        st_CudaController_get_ptr_node_info_by_index, \
        st_CudaController_get_ptr_node_info_by_node_id, \
        st_CudaController_get_ptr_node_info, \
        st_CudaController_get_ptr_node_info_by_platform_id_and_device_id, \
        st_CudaController_get_ptr_node_info_by_cuda_dev_index, \
        st_CudaController_get_ptr_node_info_by_pci_bus_id, \
        st_Cuda_get_num_all_nodes, st_Cuda_print_all_nodes, \
        st_Cuda_num_available_nodes, \
        st_Cuda_num_available_nodes_detailed, \
        st_Cuda_print_available_nodes_detailed, \
        st_Cuda_get_available_nodes_detailed, \
        st_Cuda_get_available_node_id_strs, \
        st_Cuda_get_available_node_id_strs_detailed, \
        st_NodeId_p, st_NullNodeId, st_NodeId_get_platform_id, \
        st_NodeId_get_device_id, st_NodeId_has_node_index, \
        st_NodeId_get_node_index, st_NODE_ID_STR_FORMAT_ARCHSTR

    class CudaController(NodeControllerBase):
        @staticmethod
        def NUM_ALL_NODES():
            return st_Cuda_get_num_all_nodes()

        @staticmethod
        def PRINT_ALL_NODES():
            st_Cuda_print_all_nodes()

        @staticmethod
        def NUM_AVAILABLE_NODES(filter_str=None, env_var_name=None):
            if not(filter_str is None):
                _filter_str_bytes = filter_str.strip().encode('utf-8')
                _filter_str = ct.c_char_p(filter_str)
            else:
                _filter_str = None

            if not(env_var_name is None):
                _env_var_name_bytes = env_var_name.strip().encode('utf-8')
                _env_var_name = ct.c_char_p(_env_var_name_bytes)
            else:
                _env_var_name = None

            return st_Cuda_num_available_nodes_detailed(
                _filter_str, _env_var_name)

        @staticmethod
        def PRINT_AVAILABLE_NODES(filter_str=None, env_var_name=None):
            if not(filter_str is None):
                _filter_str_bytes = filter_str.strip().encode('utf-8')
                _filter_str = ct.c_char_p(_filter_str_bytes)
            else:
                _filter_str = None

            if not(env_var_name is None):
                _env_var_name_bytes = env_var_name.strip().encode('utf-8')
                _env_var_name = ct.c_char_p(_env_var_name_bytes)
            else:
                _env_var_name = None

            st_Cuda_print_available_nodes_detailed(
                _filter_str, _env_var_name)

        @staticmethod
        def GET_AVAILABLE_NODES(filter_str=None,
                                env_var_name=None, skip_first_num_nodes=0):

            nodes = []
            if not(filter_str is None):
                _filter_str_bytes = filter_str.strip().encode('utf-8')
                _filter_str = ct.c_char_p(_filter_str_bytes)
            else:
                _filter_str = None

            if not(env_var_name is None):
                _env_var_name_bytes = env_var_name.strip().encode('utf-8')
                _env_var_name = ct.c_char_p(_env_var_name_bytes)
            else:
                _env_var_name = None

            _num_avail_nodes = st_Cuda_num_available_nodes_detailed(
                _filter_str, _env_var_name)

            if _num_avail_nodes > 0:
                node_ids_array_t = st_NodeId_p * _num_avail_nodes
                _node_ids = node_ids_array_t()
                for ii in range(0, _num_avail_nodes):
                    _node_ids[ii] = ct.cast(st_NodeId_create, st_NodeId_p)

                _num_nodes = st_Cuda_get_available_nodes_detailed(
                    _node_ids,
                    st_arch_size_t(_num_avail_nodes),
                    st_arch_size_t(skip_first_num_nodes),
                    _filter_str,
                    _env_var_name)

                for ii in range(0, _num_nodes):
                    platform_id = st_NodeId_get_platform_id(
                        ct.byref(_node_ids[ii]))
                    device_id = st_NodeId_get_platform_id(
                        ct.byref(_node_ids[ii]))
                    node_index = st_NodeId_get_node_index(
                        ct.byref(_node_ids[ii]))
                    nodes.append(
                        NodeId(
                            platform_id=platform_id,
                            device_id=device_id,
                            node_index=node_index))

                for ii in range(0, _num_avail_nodes):
                    st_NodeId_delete(_node_ids[ii])
                    _node_ids[ii] = st_NullNodeId

            return nodes

        @staticmethod
        def GET_AVAILABLE_NODE_ID_STRS(
                filter_str=None,
                env_var_name=None,
                skip_first_num_nodes=0,
                node_id_str_fmt=st_NODE_ID_STR_FORMAT_ARCHSTR.value):

            node_id_strs = []
            if not(filter_str is None):
                _filter_str_bytes = filter_str.strip().encode('utf-8')
                _filter_str = ct.c_char_p(_filter_str_bytes)
            else:
                _filter_str = None

            if not(env_var_name is None):
                _env_var_name_bytes = env_var_name.strip().encode('utf-8')
                _env_var_name = ct.c_char_p(_env_var_name_bytes)
            else:
                _env_var_name = None

            _num_avail_nodes = st_Cuda_num_available_nodes_detailed(
                _filter_str, _env_var_name)

            if _num_avail_nodes > 0:
                _node_id_str_capacity = 64
                node_id_str_buffer = [
                    ct.create_string_buffer(_node_id_str_capacity)]

                node_id_str_array_t = ct.c_char_p * _num_avail_nodes
                _tmp_node_id_strs = node_id_str_array_t()

                for ii in range(0, _num_avail_nodes):
                    _tmp_node_id_strs[ii] = ct.cast(
                        node_id_str_buffer[ii], ct.c_char_p)

                _num_node_id_strs = \
                    st_Cuda_get_available_node_id_strs_detailed(
                        _tmp_node_id_strs, st_arch_size_t(_num_avail_nodes),
                        st_arch_size_t(_node_id_str_capacity),
                        st_node_id_str_fmt_t(node_id_str_fmt),
                        st_arch_size_t(skip_first_num_nodes),
                        _filter_str, _env_var_name)

                node_id_strs = [bytes(_tmp_node_id_strs[ii]).decode('utf-8')
                                for ii in range(0, _num_node_id_strs)]
            return node_id_strs

        # *********************************************************************

        def __init__(self, config_str=None, node_id=None,
                     node_index=None, platform_id=None, device_id=None,
                     cuda_dev_index=None, **kwargs):

            if "ptr_controller" not in kwargs or \
                    kwargs["ptr_controller"] == st_NullControllerBase:
                _ptr_ctrl = st_NullCudaController

                if config_str is None or config_str == st_NullChar:
                    config_str = ''
                config_str = config_str.encode('utf-8')
                _config_str = ct.c_char_p(config_str)

                if node_id is not None and isinstance(node_id, NodeId):
                    _ptr_ctrl = st_CudaController_new_from_node_id(
                        node_id.pointer, _config_str)
                elif node_id is not None and node_id != st_NullNodeId:
                    _ptr_ctrl = st_CudaController_new_from_node_id(
                        node_id, _config_str)
                elif node_index is not None and \
                        node_index != st_NODE_UNDEFINED_INDEX.value:
                    _ptr_ctrl = st_CudaController_new_from_node_index(
                        st_node_index_t(node_index), _config_str)
                elif cuda_dev_index is not None and cuda_dev_index >= 0:
                    _ptr_ctrl = st_CudaController_new_from_cuda_device_index(
                        st_cuda_dev_index_t(cuda_dev_index), _config_str)
                elif platform_id is not None and \
                        platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                        device_id is not None and \
                        device_id != st_NODE_ILLEGAL_DEVICE_ID:
                    _ptr_ctrl = \
                        st_CudaController_new_from_platform_id_and_device_id(
                            st_node_platform_id_t(platform_id),
                            st_node_device_id_t(device_id), _config_str)
                else:
                    _ptr_ctrl = st_CudaController_new(_config_str)

                if _ptr_ctrl != st_NullCudaController:
                    kwargs["ptr_controller"] = _ptr_ctrl
                    kwargs["owns_ptr"] = True
            super().__init__(**kwargs)

        def __del__(self):
            super().__del__()

        def select_node_by_cuda_device_id(self, cuda_dev_index):
            self._last_status = \
                st_CudaController_select_node_by_cuda_device_index(
                    self._ptr_ctrl, st_cuda_dev_index_t(cuda_dev_index))

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccessful select node by cuda_dev_index op; " +
                "cuda_dev_index:{0}, status:{0}".format(
                    cuda_dev_index,
                    self._last_status))

            return self

        def select_node_by_pci_bus_id(self, pci_bus_id):
            pci_bus_id = pci_bus_id.encode('utf-8')
            self._last_status = \
                st_CudaController_select_node_by_cuda_pci_bus_id(
                    self._ptr_ctrl, ct.c_char_p(pci_bus_id))

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccessful select node by pci bus id op; " +
                "pci_bus_id:{0}, status:{0}".format(
                    pci_bus_id,
                    self._last_status))

            return self

        def managed_cobject_buffer_needs_remapping(
                self, ptr_argument, slot_size):
            return not st_CudaController_is_managed_cobject_buffer_remapped(
                self._ptr_ctrl, ptr_argument, st_buffer_size_t(slot_size))

        def remap_managed_cobject_buffer(self, ptr_argument, slot_size):
            self._last_status = st_CudaController_remap_managed_cobject_buffer(
                self._ptr_ctrl, ptr_argument, st_buffer_size_t(slot_size))

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccessful remap managed cobject buffer op; status:{0}".format(
                    self._last_status))

            return self

        def send_memory(self, ptr_argument, ptr_mem_begin, mem_size):
            self._last_status = st_CudaController_send_memory(
                self._ptr_ctrl, ptr_argument, ptr_mem_begin,
                st_arch_size_t(mem_size))

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccessful send memory op; status:{0}".format(
                    self._last_status))

            return self

        def receive_memory(self, ptr_mem_begin, ptr_argument, mem_capacity):
            self._last_status = st_CudaController_receive_memory(
                self._ptr_ctrl, ptr_mem_begin, ptr_argument,
                st_arch_size_t(mem_capacity))

            raise_error_if_status_not_success(
                self._last_status,
                "unsuccessful receive memory op; status:{0}".format(
                    self._last_status))

            return self

    from .stcommon import st_CudaArgument_p, st_NullCudaArgument, \
        st_CudaArgument_new, st_CudaArgument_new_from_buffer, \
        st_CudaArgument_new_from_raw_argument, st_CudaArgument_new_from_size, \
        st_Argument_has_cuda_arg_buffer, st_CudaArgument_get_cuda_arg_buffer, \
        st_CudaArgument_get_const_cuda_arg_buffer, \
        st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin, \
        st_CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin, \
        st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin, \
        st_CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin, \
        st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin, \
        st_CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin

    class CudaArgument(ArgumentBase):
        def __init__(self, buffer=None, ctrl=None,
                     ptr_raw_arg_begin=None, raw_arg_size=None, **kwargs):
            ptr_cuda_ctrl = st_NullCudaController
            if ctrl is not None and isinstance(ctrl, CudaController):
                ptr_cuda_ctrl = ctrl.pointer
            elif ctrl is not None and ctrl != st_NullCudaController:
                ptr_cuda_ctrl = ctrl

            if ptr_cuda_ctrl == st_NullCudaController:
                raise ValueError("creating CudaArgument requires a " +
                                 "CudaController instance")
            if "ptr_argument" not in kwargs or \
                    kwargs["ptr_argument"] == st_NullCudaArgument:
                _ptr_arg = st_NullCudaArgument
                owns_ptr = True
                ptr_buffer = st_NullBuffer
                if buffer is not None and isinstance(buffer, Buffer):
                    ptr_buffer = buffer.pointer
                else:
                    ptr_buffer = buffer
                if ptr_buffer is not None and ptr_buffer != st_NullBuffer:
                    _ptr_arg = st_CudaArgument_new_from_buffer(
                        ptr_buffer, ptr_cuda_ctrl)
                elif raw_arg_size is not None and raw_arg_size > 0:
                    _raw_arg_size = st_ctrl_size_t(raw_arg_size)
                    if ptr_raw_arg_begin is not None and \
                            ptr_raw_arg_begin != st_Null:
                        _ptr_arg = st_CudaArgument_new_from_raw_argument(
                            ptr_raw_arg_begin, _raw_arg_size, ptr_cuda_ctrl)
                    else:
                        _ptr_arg = st_CudaArgument_new_from_size(
                            _raw_arg_size, ptr_cuda_ctrl)
                else:
                    _ptr_arg = st_CudaArgument_new(ptr_cuda_ctrl)

                if _ptr_arg is not None and _ptr_arg != st_NullCudaArgument:
                    kwargs["ptr_argument"] = _ptr_arg
                    kwargs["owns_ptr"] = True
                else:
                    raise ValueError("Error during creation of CudaArgument")
            super().__init__(**kwargs)

        def __del__(self):
            super().__del__()

        @property
        def has_cuda_arg_buffer(self):
            return st_Argument_has_cuda_arg_buffer(self._ptr_argument)

        @property
        def cuda_arg_buffer(self):
            _cuda_arg_buffer = st_CudaArgument_get_cuda_arg_buffer(
                self._ptr_argument)
            return _cuda_arg_buffer

        def get_as_buffer_begin(self):
            return st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin(
                self._ptr_argument)

        def get_as_ptr_debugging_register(self):
            return st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin(
                self._ptr_argument)

        def get_as_ptr_elem_by_elem_config(self):
            return st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin(
                self._ptr_argument)

    from .buffer import Buffer
    from .trackjob import TrackJobBaseNew
    from .stcommon import st_CudaTrackJob_p, st_NullCudaTrackJob, \
        st_CudaTrackJob_create, st_CudaTrackJob_new_from_config_str, \
        st_CudaTrackJob_new, st_CudaTrackJob_new_with_output, \
        st_CudaTrackJob_new_detailed, st_CudaTrackJob_has_controller, \
        st_CudaTrackJob_get_ptr_controller, st_CudaTrackJob_has_particles_arg, \
        st_CudaTrackJob_get_ptr_particles_arg, \
        st_CudaTrackJob_has_beam_elements_arg, \
        st_CudaTrackJob_get_ptr_beam_elements_arg, \
        st_CudaTrackJob_has_output_arg, st_CudaTrackJob_get_ptr_output_arg, \
        st_CudaTrackJob_has_elem_by_elem_config_arg, \
        st_CudaTrackJob_get_ptr_elem_by_elem_config_arg, \
        st_CudaTrackJob_has_debug_register_arg, \
        st_CudaTrackJob_get_ptr_debug_register_arg, \
        st_CudaTrackJob_has_particles_addr_arg, \
        st_CudaTrackJob_get_ptr_particles_addr_arg, \
        st_Controller_select_node

    class CudaTrackJob(TrackJobBaseNew):
        def __init__(self, beam_elements_buffer, particles_buffer,
                     particle_set_index=0, until_turn_elem_by_elem=0,
                     output_buffer=None, node_id_str=None, config_str=None):
            _ptr_track_job = st_NullCudaTrackJob

            if config_str is not None and config_str != "":
                _config_str = ct.c_char_p(config_str.encode('utf-8'))
                _ptr_track_job = st_CudaTrackJob_new_from_config_str(
                    _config_str)
            else:
                _ptr_track_job = st_CudaTrackJob_create()

            if _ptr_track_job == st_NullCudaTrackJob:
                raise RuntimeError("Error while creating CudaTrackJob")

            _last_status = st_ARCH_STATUS_SUCCESS.value

            _ptr_cuda_ctrl = \
                st_CudaTrackJob_get_ptr_controller(_ptr_track_job)

            if not st_CudaTrackJob_has_controller(_ptr_track_job) or \
                    _ptr_cuda_ctrl is None or \
                    _ptr_cuda_ctrl == st_NullCudaController:
                raise RuntimeError("CudaTrackJob requires CudaController")

            if node_id_str is not None and node_id_str != "":
                _node_id_str = ct.c_char_p(node_id_str.encode('utf-8'))
                _last_status = st_Controller_select_node(
                    _ptr_cuda_ctrl, _node_id_str)
                raise_error_if_status_not_success(
                    _last_status,
                    "Error selection of node by node_id_str {0} " +
                    "status:{1}".format(
                        node_id_str,
                        _last_status))

            if _last_status == st_ARCH_STATUS_SUCCESS.value:
                super().__init__(ptr_track_job=_ptr_track_job, owns_ptr=True)

            if self._ptr_track_job is None or \
                    self._ptr_track_job == st_NullCudaTrackJob or \
                    self._last_status != st_ARCH_STATUS_SUCCESS.value:
                raise RuntimeError("Error during creation of TrackJobBase")

            super()._reset_detailed(
                beam_elements_buffer,
                particles_buffer,
                particle_set_index,
                until_turn_elem_by_elem,
                output_buffer)

            if self._last_status != st_ARCH_STATUS_SUCCESS.value:
                raise RuntimeError("Error during resetting TrackJobBase")

        def __del__(self):
            super().__del__()

        @property
        def has_particles_arg(self):
            return st_CudaTrackJob_has_particles_arg(self._ptr_track_job)

        @property
        def particles_arg(self):
            _ptr_argument = st_CudaTrackJob_get_ptr_particles_arg(
                self._ptr_track_job)
            return CudaArgument(ptr_argument=_ptr_argument, owns_ptr=False)

        @property
        def has_beam_elements_arg(self):
            return st_CudaTrackJob_has_beam_elements_arg(self._ptr_track_job)

        @property
        def beam_elements_arg(self):
            _ptr_argument = st_CudaTrackJob_get_ptr_beam_elements_arg(
                self._ptr_track_job)
            return CudaArgument(ptr_argument=_ptr_argument, owns_ptr=False)

        @property
        def has_output_arg(self):
            return st_CudaTrackJob_has_output_arg(self._ptr_track_job)

        @property
        def output_arg(self):
            _ptr_argument = st_CudaTrackJob_get_ptr_output_arg(
                self._ptr_track_job)
            return CudaArgument(ptr_argument=_ptr_argument, owns_ptr=False)

        @property
        def has_particles_addr_arg(self):
            return st_CudaTrackJob_has_particles_addr_arg(self._ptr_track_job)

        @property
        def particles_addr_arg(self):
            _ptr_argument = st_CudaTrackJob_get_ptr_particles_addr_arg(
                self._ptr_track_job)
            return CudaArgument(ptr_argument=_ptr_argument, owns_ptr=False)

        @property
        def has_debug_register_arg(self):
            return st_CudaTrackJob_has_debug_register_arg(self._ptr_track_job)

        @property
        def particles_arg(self):
            _ptr_argument = st_CudaTrackJob_get_ptr_debug_register_arg(
                self._ptr_track_job)
            return CudaArgument(ptr_argument=_ptr_argument, owns_ptr=False)

        @property
        def has_controller(self):
            return st_CudaTrackJob_has_controller(self._ptr_track_job)

        @property
        def controller(self):
            _ptr_cuda_ctrl = st_CudaTrackJob_get_ptr_controller(
                self._ptr_track_job)
            return CudaController(
                ptr_controller=_ptr_cuda_ctrl,
                owns_ptr=False)


else:
    class CudaNodeInfo(object):
        pass

    class CudaController(object):
        @staticmethod
        def NUM_ALL_NODES():
            raise RuntimeError("Cuda module disabled, no nodes present")
            return 0

        @staticmethod
        def PRINT_ALL_NODES():
            raise RuntimeError("Cuda module disabled, no nodes to print")
            st_Cuda_print_all_nodes()

        @staticmethod
        def NUM_AVAILABLE_NODES(filter_str=None, env_var_name=None):
            raise RuntimeError("Cuda module disabled, no nodes available")
            return 0

        @staticmethod
        def PRINT_AVAILABLE_NODES(filter_str=None, env_var_name=None):
            raise RuntimeError("Cuda module disabled, no nodes to print")

        @staticmethod
        def GET_AVAILABLE_NODES(filter_str=None,
                                env_var_name=None, skip_first_num_nodes=0):
            raise RuntimeError("Cuda module disabled, no nodes available")
            return []

        @staticmethod
        def GET_AVAILABLE_NODE_ID_STRS(
                filter_str=None,
                env_var_name=None,
                skip_first_num_nodes=0,
                node_id_str_fmt=st_NODE_ID_STR_FORMAT_ARCHSTR.value):
            raise RuntimeError("Cuda module disabled, no nodes available")
            return []

    class CudaArgument(object):
        pass

    class CudaTrackJob(object):
        pass
