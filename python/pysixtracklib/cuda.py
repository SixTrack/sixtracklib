#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from cobjects import CBuffer, CObject
from .particles import ParticlesSet
from .config import SIXTRACKLIB_MODULES
from .trackjob import _get_buffer

if SIXTRACKLIB_MODULES.get('cuda', False):
    from .stcommon import st_CudaContext_p, st_NullCudaContext, st_Null, \
        st_CudaArgument_p, st_NullCudaArgument, st_Buffer_p, st_NullBuffer, \
        st_CudaContext_create, st_CudaContext_delete, \
        st_CudaArgument_new, st_CudaArgument_delete, \
        st_CudaArgument_send_buffer, st_CudaArgument_send_memory, \
        st_CudaArgument_receive_buffer, st_CudaArgument_receive_memory, \
        st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete

    class CudaContext(object):
        def __init__(self, device=None, config_str=None):
            self._ptr_cuda_context = st_CudaContext_create()

        def __del__(self):
            if self._ptr_cuda_context != st_NullCudaContext:
                st_CudaContext_delete(self._ptr_cuda_context)
                self._ptr_cuda_context = st_NullCudaContext

        @property
        def pointer(self):
            return self._ptr_cuda_context

    class CudaArgument(object):
        def __init__(self, ctx, user_arg, length=0):
            self._mode = None
            self._ptr_user_arg = st_Null
            self._ptr_cuda_arg = st_NullCudaArgument
            self._arg_length = ct.c_uint64(0)
            self._status = -1

            try:
                buffer = _get_buffer(user_arg)
            except ValueError:
                buffer = None

            if buffer is not None:
                self._ptr_user_arg = st_Buffer_new_mapped_on_cbuffer(buffer)
                self._mode = "buffer"
            elif length and length > 0:
                self._ptr_user_arg = user_arg
                self._arg_length = ct.c_uint64(length)
                self._mode = "raw"
            else:
                raise ValueError("needs either a CBuffer or a pointer + " +
                                 "length as input parameters")

            if self._ptr_user_arg != st_Null and self._mode is not None:
                self._ptr_cuda_arg = st_CudaArgument_new(ctx.pointer)

            if self._ptr_cuda_arg != st_NullCudaArgument:
                if self._mode == 'buffer':
                    self._status = st_CudaArgument_send_buffer(
                        self._ptr_cuda_arg, self._ptr_user_arg)
                elif self._mode == 'raw':
                    self._arg_length = ct.c_uint64(length)
                    self._status = st_CudaArgument_send_memory(
                        self._ptr_cuda_arg, self._ptr_user_arg,
                        self._arg_length)

        def __del__(self):
            if self._mode == 'buffer':
                if self._ptr_user_arg != st_NullBuffer:
                    st_Buffer_delete(self._ptr_user_arg)
                    self._ptr_user_arg = st_NullBuffer

            if self._ptr_cuda_arg != st_NullCudaArgument:
                st_CudaArgument_delete(self._ptr_cuda_arg)
                self._ptr_cuda_arg = st_NullCudaArgument

        def send(self):
            assert self._ptr_cuda_arg != st_NullCudaArgument
            if self._mode == 'buffer' and self._ptr_user_arg != st_NullBuffer:
                self._status = st_CudaArgument_send_buffer(
                    self._ptr_cuda_arg, self._ptr_user_arg)
            elif self._mode == 'raw' and self._ptr_user_arg != st_Null and \
                    self._arg_length.value > 0:
                self._status = st_CudaArgument_send_memory(
                    self._ptr_cuda_arg, self._ptr_user_arg, self._arg_length)
            else:
                self._status = -1
            return self

        def receive(self):
            assert self._ptr_cuda_arg != st_NullCudaArgument
            if self._mode == 'buffer' and self._ptr_user_arg != st_NullBuffer:
                self._status = st_CudaArgument_receive_buffer(
                    self._ptr_cuda_arg, self._ptr_user_arg)
            elif self._mode == 'raw' and self._ptr_user_arg != st_Null and \
                    self._arg_length.value > 0:
                self._status = st_CudaArgument_receive_memory(
                    self._ptr_cuda_arg, self._ptr_user_arg, self._arg_length)
            else:
                self._status = -1
            return self

        @property
        def last_status(self):
            return self._status

else:

    class CudaContext(object):
        pass

    class CudaArgument(object):
        pass
