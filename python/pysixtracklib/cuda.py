#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer, CObject
from .particles import ParticlesSet
from .config import SIXTRACKLIB_MODULES
from .control import NodeId, NodeInfoBase, ArgumentBase, NodeControllerBase

if SIXTRACKLIB_MODULES.get('cuda', False):
    from .stcommon import st_Null, st_node_platform_id_t, st_node_device_id_t, \
        st_node_index_t, st_arch_status_t, st_arch_size_t, st_buffer_size_t, \
        st_ctrl_size_t, st_ctrl_status_t, \
        st_NullChar, st_ARCH_STATUS_GENERAL_FAILURE, st_ARCH_STATUS_SUCCESS, \
        st_NODE_UNDEFINED_INDEX, st_NODE_ILLEGAL_PLATFORM_ID, \
        st_NODE_ILLEGAL_DEVICE_ID, st_NullNodeId

    from .stcommon import st_CudaNodeInfo_p, st_NullCudaNodeInfo, \
        st_cuda_dev_index_t, st_CudaNodeInfo_new, st_CudaNodeInfo_new_detailed,\
        st_CudaNodeInfo_get_cuda_device_index, \
        st_CudaNodeInfo_get_pci_bus_id_str, st_CudaNodeInfo_get_warp_size, \
        st_CudaNodeInfo_get_compute_capability, \
        st_CudaNodeInfo_get_num_multiprocessors, \
        st_CudaNodeInfo_get_max_threads_per_block, \
        st_CudaNodeInfo_get_max_threads_per_multiprocessor

    class CudaNodeInfo(NodeInfoBase):
        def __init__( self, ptr_node_info=st_NullCudaNodeInfo, owns_ptr=False,
                      cuda_dev_index=None, platform_id=None,
                      device_id=None, node_index=None, is_default_node=False,
                      is_selected_node=False ):
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
                        st_cuda_dev_index_t( cuda_dev_index ),
                        st_node_platform_id_t( platform_id ),
                        st_node_device_id_t( device_id ),
                        st_node_index_t( node_index ),
                        ct.c_bool( is_default_node ),
                        ct.c_bool( is_selected_node ) )

                    owns_ptr = bool( _ptr_node_info != st_NullCudaNodeInfo )

            if _ptr_node_info != st_NullCudaNodeInfo:
                super().__init__(
                    ptr_node_info=_ptr_node_info, owns_ptr=owns_ptr )

        @property
        def cuda_device_index(self):
            return st_CudaNodeInfo_get_cuda_device_index( self._ptr_node_info )

        @property
        def pci_bus_id(self):
            _pci_bus_id_str = None
            _ptr_bus_id_cstr = st_CudaNodeInfo_get_pci_bus_id_str(
                self._ptr_node_info )
            if _ptr_bus_id_cstr != st_NullChar:
                _pci_bus_id_str = str( _ptr_bus_id_cstr.value )
                _pci_bus_id_str = _pci_bus_id_str.decode( 'utf-8' )
            return _pci_bus_id_str

        @property
        def warp_size(self):
            return st_CudaNodeInfo_get_warp_size( self._ptr_node_info )

        @property
        def compute_capability(self):
            return st_CudaNodeInfo_get_compute_capability(
                self._ptr_node_info )

        @property
        def num_multiprocessors(self):
            return st_CudaNodeInfo_get_num_multiprocessors(
                self._ptr_node_info )

        @property
        def max_threads_per_block(self):
            return st_CudaNodeInfo_get_max_threads_per_block(
                self._ptr_node_info )

        @property
        def max_threads_per_multiprocessor(self):
            st_CudaNodeInfo_get_max_threads_per_multiprocessor(
                self._ptr_node_info )

    from .stcommon import st_CudaController_p, st_NullCudaController, \
        st_CudaController_create, st_CudaController_new, \
        st_CudaController_new_from_node_id, \
        st_CudaController_new_from_platform_id_and_device_id, \
        st_CudaController_new_from_node_index, \
        st_CudaController_new_from_cuda_device_index, \
        st_CudaController_select_node_by_cuda_device_index, \
        st_CudaController_select_node_by_cuda_pci_bus_id, \
        st_CudaController_remap_managed_cobject_buffer, \
        st_CudaController_is_managed_cobject_buffer_remapped, \
        st_CudaController_send_memory, st_CudaController_receive_memory, \
        st_CudaController_get_ptr_node_info_by_index, \
        st_CudaController_get_ptr_node_info_by_node_id, \
        st_CudaController_get_ptr_node_info, \
        st_CudaController_get_ptr_node_info_by_platform_id_and_device_id, \
        st_CudaController_get_ptr_node_info_by_cuda_dev_index, \
        st_CudaController_get_ptr_node_info_by_pci_bus_id


    class CudaController(NodeControllerBase):
        def __init__(self, config_str=None, node_id=None,
                     node_index=None, platform_id=None, device_id=None,
                     cuda_dev_index=None ):

            _ptr_ctrl = st_NullCudaController

            if config_str is None or config_str == st_NullChar:
                config_str = ''
            config_str = config_str.encode( 'utf-8' )
            _config_str = ct.c_char_p( config_str )

            if node_id is not None and isinstance( node_id, NodeId ):
                _ptr_ctrl = st_CudaController_new_from_node_id(
                    node_id.pointer, _config_str )
            elif node_id is not None and node_id != st_NullNodeId:
                _ptr_ctrl = st_CudaController_new_from_node_id(
                    node_id, _config_str )
            elif node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
                _ptr_ctrl = st_CudaController_new_from_node_index(
                    st_node_index_t( node_index ), _config_str )
            elif cuda_dev_index is not None and cuda_dev_index >= 0:
                _ptr_ctrl = st_CudaController_new_from_cuda_device_index(
                    st_cuda_dev_index_t( cuda_dev_index ), _config_str )
            elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID:
                _ptr_ctrl = \
                    st_CudaController_new_from_platform_id_and_device_id(
                        st_node_platform_id_t( platform_id ),
                        st_node_device_id_t( device_id ), _config_str )
            else:
                _ptr_ctrl = st_CudaController_new( _config_str )

            if _ptr_ctrl != st_NullCudaController:
                super().__init__( ptr_controller=_ptr_ctrl, owns_ptr=True )
            else:
                raise ValueError( "unable to create CudaController C-pointer" )

        def select_node_by_cuda_device_id( self, cuda_dev_index ):
            self._last_status = \
                st_CudaController_select_node_by_cuda_device_index(
                    self._ptr_ctrl, st_cuda_dev_index_t( cuda_dev_index ) )
            return self

        def select_node_by_pci_bus_id( self, pci_bus_id ):
            pci_bus_id = pci_bus_id.encode( 'utf-8' )
            self._last_status = \
                st_CudaController_select_node_by_cuda_pci_bus_id(
                    self._ptr_ctrl, ct.c_char_p( pci_bus_id ) )
            return self

        def remap_managed_cobject_buffer( self, ptr_argument, slot_size ):
            self._last_status = st_CudaController_remap_managed_cobject_buffer(
                self._ptr_ctrl, ptr_argument, st_buffer_size_t( slot_size ) )
            return self

        def is_managed_cobject_buffer_remapped( self, ptr_argument, slot_size):
            return st_CudaController_is_managed_cobject_buffer_remapped(
                self._ptr_ctrl, ptr_argument, st_buffer_size_t( slot_size ) )

        def send_memory( self, ptr_argument, ptr_mem_begin, mem_size ):
            self._last_status = st_CudaController_send_memory(
                self._ptr_ctrl, ptr_argument, ptr_mem_begin,
                    st_arch_size_t( mem_size ) )
            return self

        def receive_memory( self, ptr_mem_begin, ptr_argument, mem_capacity ):
            self._last_status = st_CudaController_receive_memory(
                self._ptr_ctrl, ptr_mem_begin, ptr_argument,
                    st_arch_size_t( mem_capacity ) )
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
        def __init__( self, ptr_buffer=None, ptr_cuda_ctrl=st_NullCudaController,
                      ptr_raw_arg_begin=None, raw_arg_size=None,
                      ptr_argument=st_NullCudaArgument, owns_ptr=True ):
            if ptr_cuda_ctrl is None or ptr_cuda_ctrl == st_NullCudaController:
                raise ValueError( "creating CudaArgument requires a " + \
                                  "CudaController instance" )

            _ptr_arg = ptr_argument
            if _ptr_arg == st_NullCudaArgument or _ptr_arg is None:
                owns_ptr = True
                if ptr_buffer is not None and ptr_buffer != st_NullBuffer:
                    _ptr_arg = st_CudaArgument_new_from_buffer(
                        ptr_buffer, ptr_cuda_ctrl )
                elif raw_arg_size is not None and raw_arg_size > 0:
                    _raw_arg_size = st_ctrl_size_t( raw_arg_size )
                    if ptr_raw_arg_begin is not None and \
                        ptr_raw_arg_begin != st_Null:
                        _ptr_arg = st_CudaArgument_new_from_raw_argument(
                            ptr_raw_arg_begin, _raw_arg_size, ptr_cuda_ctrl )
                    else:
                        _ptr_arg = st_CudaArgument_new_from_size(
                            _raw_arg_size, ptr_cuda_ctrl )
                else:
                    _ptr_arg = st_CudaArgument_new( ptr_cuda_ctrl )

            if _ptr_arg is not None and _ptr_arg != st_NullCudaArgument:
                super().__init__( ptr_argument=_ptr_arg, owns_ptr=owns_ptr )
            else:
                raise ValueError( "unable to create CudaArgument C-pointer" )

        @property
        def has_cuda_arg_buffer(self):
            return st_Argument_has_cuda_arg_buffer( self._ptr_argument )

        @property
        def cuda_arg_buffer(self):
            return st_CudaArgument_get_cuda_arg_buffer( self._ptr_argument )

        def get_as_buffer_begin( self ):
            return st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin(
                self._ptr_argument )

        def get_as_ptr_debugging_register( self ):
            return st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin(
                self._ptr_argument )

        def get_as_ptr_elem_by_elem_config( self ):
            return st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin(
                self._ptr_argument )

else:
    class CudaNodeInfo(object):
        pass

    class CudaController(object):
        pass

    class CudaArgument(object):
        pass
