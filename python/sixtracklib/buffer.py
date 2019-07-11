#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cobjects
from cobjects import CBuffer

import ctypes as ct

from .stcommon import st_Buffer, st_Null, st_NullChar, st_NullUChar, \
    st_ARCH_STATUS_GENERAL_FAILURE, st_ARCH_STATUS_SUCCESS, st_arch_status_t, \
    st_buffer_size_t, st_arch_size_t, st_uchar_p, st_buffer_addr_t, \
    st_Buffer, st_Buffer_p, st_NullBuffer, st_Buffer_preset, \
    st_Buffer_new, st_Buffer_new_on_memory, st_Buffer_new_on_data, \
    st_Buffer_new_from_file, st_Buffer_init_from_data, st_Buffer_get_slot_size,\
    st_Buffer_get_num_of_objects, st_Buffer_get_size, st_Buffer_get_capacity, \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_get_header_size, \
    st_Buffer_get_data_begin_addr, st_Buffer_get_data_end_addr, \
    st_Buffer_get_objects_begin_addr, st_Buffer_get_objects_end_addr, \
    st_Buffer_get_num_of_slots, st_Buffer_get_max_num_of_slots, \
    st_Buffer_get_num_of_objects, st_Buffer_get_max_num_of_objects, \
    st_Buffer_get_num_of_dataptrs, st_Buffer_get_max_num_of_dataptrs, \
    st_Buffer_get_num_of_garbage_ranges, st_Buffer_get_max_num_of_garbage_ranges, \
    st_Buffer_read_from_file, st_Buffer_write_to_file, \
    st_Buffer_write_to_file_normalized_addr, st_Buffer_reset, \
    st_Buffer_reset_detailed, st_Buffer_reserve, st_Buffer_reserve_capacity, \
    st_Buffer_needs_remapping, st_Buffer_remap, st_Buffer_refresh, \
    st_Buffer_clear, st_Buffer_free, st_Buffer_delete


class Buffer(object):
    def __init__(self, size=0, ptr_data=st_NullUChar, path_to_file=None,
                 cbuffer=None, ptr_ext_buffer=st_NullBuffer, orig=None,
                 owns_ptr=True, slot_size=8):
        self._ptr_st_buffer = st_NullBuffer
        self._last_status = st_ARCH_STATUS_SUCCESS.value
        self._on_del = "delete"

        if ptr_ext_buffer is not None and ptr_ext_buffer != st_NullBuffer:
            self._ptr_st_buffer = ptr_ext_buffer
            if owns_ptr is None or not(owns_ptr is True):
                self._on_del = None
        elif cbuffer is not None and isinstance(cbuffer, CBuffer):
            self._ptr_st_buffer = st_Buffer_new_mapped_on_cbuffer(cbuffer)
        elif path_to_file is not None and path_to_file != "":
            _path_to_file = ct.c_char_p(path_to_file.encode('utf-8'))
            self._ptr_st_buffer = st_Buffer_new_from_file(_path_to_file)
        elif ptr_data is not None and ptr_data != st_NullUChar and \
                slot_size is not None and slot_size > 0:
            self._ptr_st_buffer = st_Buffer_new_on_data(
                ptr_data, st_buffer_size_t(slot_size))
        elif size is not None and size >= 0:
            self._ptr_st_buffer = st_Buffer_new(st_buffer_size_t(size))
        else:
            self._ptr_st_buffer = st_Buffer_new(st_buffer_size_t(0))

    def __del__(self):
        if self._ptr_st_buffer != st_NullBuffer and self._on_del is not None:
            if self._on_del == "delete":
                st_Buffer_delete(self._ptr_st_buffer)
                self._ptr_st_buffer = st_NullBuffer
            elif self._on_del == "free":
                st_Buffer_free(self._ptr_st_buffer)
                st_Buffer_preset(self._ptr_st_buffer)

    @property
    def pointer(self):
        return self._ptr_st_buffer

    @property
    def last_status(self):
        return self._last_status.value

    @property
    def last_status_success(self):
        return self._last_status == st_ARCH_STATUS_SUCCESS.value

    @property
    def slot_size(self):
        return st_Buffer_get_slot_size(self._ptr_st_buffer)

    @property
    def size(self):
        return st_Buffer_get_size(self._ptr_st_buffer)

    @property
    def capacity(self):
        return st_Buffer_get_capacity(self._ptr_st_buffer)

    @property
    def num_objects(self):
        return st_Buffer_get_num_of_objects(self._ptr_st_buffer)

    @property
    def max_num_objects(self):
        return st_Buffer_get_max_num_of_objects(self._ptr_st_buffer)

    @property
    def num_slots(self):
        return st_Buffer_get_num_of_slots(self._ptr_st_buffer)

    @property
    def max_num_slots(self):
        return st_Buffer_get_max_num_of_slots(self._ptr_st_buffer)

    @property
    def num_dataptrs(self):
        return st_Buffer_get_num_of_dataptrs(self._ptr_st_buffer)

    @property
    def max_num_dataptrs(self):
        return st_Buffer_get_max_num_of_dataptrs(self._ptr_st_buffer)

    @property
    def num_garbage_ranges(self):
        return st_Buffer_get_num_of_garbage_ranges(self._ptr_st_buffer)

    @property
    def max_num_garbage_ranges(self):
        return st_Buffer_get_max_num_of_garbage_ranges(self._ptr_st_buffer)

    @property
    def needs_remapping(self):
        return st_Buffer_needs_remapping(self._ptr_st_buffer)

    @property
    def header_size(self):
        return st_Buffer_get_header_size(self._ptr_st_buffer)

    @property
    def data_begin_addr(self):
        return st_Buffer_get_data_begin_addr(self._ptr_st_buffer)

    @property
    def data_end_addr(self):
        return st_Buffer_get_data_end_addr(self._ptr_st_buffer)

    @property
    def ptr_data_begin(self):
        return ct.cast(st_Buffer_get_data_begin_addr(
            self._ptr_st_buffer), st_uchar_p)

    @property
    def ptr_data_end(self):
        return ct.cast(st_Buffer_get_data_end_addr(
            self._ptr_st_buffer), st_uchar_p)

    def clear(self, clear_data=False):
        self._last_status = st_Buffer_clear(
            self._ptr_st_buffer, ct.c_bool(clear_data is True))
        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("unsuccessful clear op; status:{0}".format(
                self._last_status))
        return self

    def refresh(self):
        self._last_status = st_Buffer_refresh(self._ptr_st_buffer)
        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("unsuccessful refresh op; status:{0}".format(
                self._last_status))
        return self

    def remap(self):
        self._last_status = st_Buffer_remap(self._ptr_st_buffer)
        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("unsuccessful reamp op; status:{0}".format(
                self._last_status))
        return self

    def write_to_file(self, path_to_file, normalized_base_addr=None):
        if path_to_file is not None and path_to_file != "":
            _path_to_file = ct.c_char_p(path_to_file.encode('utf-8'))
            if normalized_base_addr is not None and normalized_base_addr > 0:
                self._last_status = st_Buffer_write_to_file_normalized_addr(
                    self._ptr_st_buffer, _path_to_file, st_buffer_addr_t(
                        normalized_base_addr))
            else:
                self._last_status = st_Buffer_write_to_file(
                    self._ptr_st_buffer, _path_to_file)

            if self._last_status != st_ARCH_STATUS_SUCCESS.value:
                raise RuntimeError("unsuccessful write_to_file op; " +
                                   "status:{0}".format(self._last_status))
        else:
            self._last_status = st_ARCH_STATUS_GENERAL_FAILURE
            raise RuntimeError("illegal path supplied")

        return self

    def read_from_file(self, path_to_file):
        if path_to_file is not None and path_to_file != "":
            self._last_status = st_Buffer_read_from_file(
                self._ptr_st_buffer, ct.c_char_p(
                    path_to_file.encode('utf-8')))
            if self._last_status != st_ARCH_STATUS_SUCCESS.value:
                raise RuntimeError("unsuccessful read_from_file op; " +
                                   "status:{0}".format(self._last_status))
        return self

    def reserve_capacity(self, capacity):
        if capacity is not None and capacity >= 0:
            self._last_status = st_Buffer_reserve_capacity(
                self._ptr_st_buffer, st_buffer_size_t(capacity))
            if self._last_status != st_ARCH_STATUS_SUCCESS.value:
                raise RuntimeError("unsuccessful reserve capacity op; " +
                                   "status:{0}".format(self._last_status))
        return self

    def reserve(self, max_num_objects, max_num_slots, max_num_dataptrs,
                max_num_garbage_ranges):
        self._last_status = st_Buffer_reserve(
            self._ptr_st_buffer,
            max_num_objects,
            max_num_slots,
            max_num_dataptrs,
            max_num_garbage_ranges)

        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("unsuccessful reserve op; status:{0}".format(
                self._last_status))

        return self

    def reset(self, max_num_objects=None, max_num_slots=None,
              max_num_dataptrs=None, max_num_garbage_ranges=None):
        if max_num_slots is None and max_num_objects is None and \
                max_num_dataptrs is None and max_num_garbage_ranges is None:
            self._last_status = st_Buffer_reset(self._ptr_st_buffer)
        else:
            if max_num_objects is None:
                max_num_objects = self.max_num_objects
            if max_num_slots is None:
                max_num_slots = self.max_num_slots
            if max_num_dataptrs is None:
                max_num_dataptrs = self.max_num_dataptrs
            if max_num_garbage_ranges is None:
                max_num_garbage_ranges = self.max_num_garbage_ranges

            self._last_status = st_Buffer_reset_detailed(
                self._ptr_st_buffer, st_buffer_size_t(max_num_objects),
                st_buffer_size_t(max_num_slots),
                st_buffer_size_t(max_num_dataptrs),
                st_buffer_size_t(max_num_garbage_ranges))

        if self._last_status != st_ARCH_STATUS_SUCCESS.value:
            raise RuntimeError("unsuccessful reset op; status:{0}".format(
                self._last_status))
        return self


# end: python/sixtracklib/buffer.py
