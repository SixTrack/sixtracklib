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
        st_CudaController_get_ptr_node_info_by_platform_id_and_device_id
                

    class CudaController(NodeControllerBase):
        def __init__(self, config_str=None, node_id=None, 
                     node_index=None, platform_id=None, device_id=None, 
                     cuda_dev_index=None ):
            _ptr_ctrl = st_NullCudaController
            if config_str is not None and config_str != "":
                config_str = config_str.encode( 'utf-8' )                
                _ptr_ctrl = st_CudaController_new( ct.c_char_p( config_str ) )
            elif node_id is not None and isinstance( node_id, NodeId ):
                _ptr_ctrl = st_CudaController_new_from_node_id( 
                    node_id.pointer )
            elif node_id is not None and node_id != st_NullNodeId:
                _ptr_ctrl = st_CudaController_new_from_node_id( node_id )
            elif node_index is not None and \
                node_index != st_NODE_UNDEFINED_INDEX.value:
                _ptr_ctrl = st_CudaController_new_from_node_index(
                    st_node_index_t( node_index ) )
            elif cuda_dev_index is not None and cuda_dev_index > 0:
                _ptr_ctrl = st_CudaController_new_from_cuda_device_index(
                    ct.c_int( cuda_dev_index ) )
            elif platform_id is not None and \
                platform_id != st_NODE_ILLEGAL_PLATFORM_ID.value and \
                device_id is not None and \
                device_id != st_NODE_ILLEGAL_DEVICE_ID:
                _ptr_ctrl = \
                    st_CudaController_new_from_platform_id_and_device_id(
                        st_node_platform_id_t( platform_id ),
                        st_node_device_id_t( device_id ) )
            else:
                _ptr_ctrl = st_CudaController_create()
                
            if _ptr_ctrl != st_NullCudaController:
                super().__init__( ptr_controller=_ptr_ctrl, owns_ptr=True )
            else:
                raise ValueError( "unable to create CudaController C-pointer" )
            
        def select_node_by_cuda_device_id( self, cuda_dev_index ):
            self._last_status = \
                st_CudaController_select_node_by_cuda_device_index(
                    self._ptr_ctrl, ct.c_int( cuda_dev_index ) )
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
        def __init__( self, ptr_cuda_ctrl=st_NullCudaController, 
                      ptr_argument=st_NullCudaArgument, owns_ptr=True,
                      ptr_buffer=None, ptr_raw_arg_begin=None, raw_arg_size ):
            
        
            
    
    
    
        

#    class CudaArgument(object):
#        def __init__(self, ctx, user_arg, length=0):
#            self._mode = None
#            self._ptr_user_arg = st_Null
#            self._ptr_cuda_arg = st_NullCudaArgument
#            self._arg_length = ct.c_uint64(0)
#            self._status = -1
#
#            try:
#                buffer = _get_buffer(user_arg)
#            except ValueError:
#                buffer = None
#
#            if buffer is not None:
#                self._ptr_user_arg = st_Buffer_new_mapped_on_cbuffer(buffer)
#                self._mode = "buffer"
#            elif length and length > 0:
#                self._ptr_user_arg = user_arg
#                self._arg_length = ct.c_uint64(length)
#                self._mode = "raw"
#            else:
#                raise ValueError("needs either a CBuffer or a pointer + " +
#                                 "length as input parameters")
#
#            if self._ptr_user_arg != st_Null and self._mode is not None:
#                self._ptr_cuda_arg = st_CudaArgument_new(ctx.pointer)
#
#            if self._ptr_cuda_arg != st_NullCudaArgument:
#                if self._mode == 'buffer':
#                    self._status = st_CudaArgument_send_buffer(
#                        self._ptr_cuda_arg, self._ptr_user_arg)
#                elif self._mode == 'raw':
#                    self._arg_length = ct.c_uint64(length)
#                    self._status = st_CudaArgument_send_memory(
#                        self._ptr_cuda_arg, self._ptr_user_arg,
#                        self._arg_length)
#
#        def __del__(self):
#            if self._mode == 'buffer':
#                if self._ptr_user_arg != st_NullBuffer:
#                    st_Buffer_delete(self._ptr_user_arg)
#                    self._ptr_user_arg = st_NullBuffer
#
#            if self._ptr_cuda_arg != st_NullCudaArgument:
#                st_CudaArgument_delete(self._ptr_cuda_arg)
#                self._ptr_cuda_arg = st_NullCudaArgument
#
#        def send(self):
#            assert self._ptr_cuda_arg != st_NullCudaArgument
#            if self._mode == 'buffer' and self._ptr_user_arg != st_NullBuffer:
#                self._status = st_CudaArgument_send_buffer(
#                    self._ptr_cuda_arg, self._ptr_user_arg)
#            elif self._mode == 'raw' and self._ptr_user_arg != st_Null and \
#                    self._arg_length.value > 0:
#                self._status = st_CudaArgument_send_memory(
#                    self._ptr_cuda_arg, self._ptr_user_arg, self._arg_length)
#            else:
#                self._status = -1
#            return self
#
#        def receive(self):
#            assert self._ptr_cuda_arg != st_NullCudaArgument
#            if self._mode == 'buffer' and self._ptr_user_arg != st_NullBuffer:
#                self._status = st_CudaArgument_receive_buffer(
#                    self._ptr_cuda_arg, self._ptr_user_arg)
#            elif self._mode == 'raw' and self._ptr_user_arg != st_Null and \
#                    self._arg_length.value > 0:
#                self._status = st_CudaArgument_receive_memory(
#                    self._ptr_cuda_arg, self._ptr_user_arg, self._arg_length)
#            else:
#                self._status = -1
#            return self
#
#        @property
#        def last_status(self):
#            return self._status

else:

    class CudaController(object):
        pass

    #class CudaArgument(object):
        #pass
