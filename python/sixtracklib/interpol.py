#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import ctypes as ct
from cobjects import CBuffer, CObject, CField

from .trackjob import TrackJob
from .cuda import CudaTrackJob
from .beam_elements import Elements, SpaceChargeInterpolatedProfile
from .buffer import AssignAddressItem, get_cbuffer_from_obj, Buffer
from .stcommon import st_ARCH_STATUS_SUCCESS, \
    st_ARCH_ILLEGAL_BUFFER_ID, \
    st_ARCH_BEAM_ELEMENTS_BUFFER_ID, \
    st_LineDensityProfileData_p, \
    st_LineDensityProfileData_type_id, \
    st_NullLineDensityProfileData, \
    st_NullSpaceChargeInterpolatedProfile, \
    st_SpaceChargeInterpolatedProfile_type_id, \
    st_SpaceChargeInterpolatedProfile_interpol_data_addr_offset, \
    st_LineDensityProfileData_values_offset, \
    st_LineDensityProfileData_derivatives_offset, \
    st_LineDensityProfileData_prepare_interpolation, \
    st_LineDensityProfileData_interpolate_value, \
    st_LineDensityProfileData_interpolate_1st_derivative, \
    st_LineDensityProfileData_interpolate_2nd_derivative

class LineDensityProfileData(CObject):
    _typeid = st_LineDensityProfileData_type_id()

    INTERPOL_LINEAR = 0
    INTERPOL_CUBIC = 1
    INTERPOL_NONE = 255

    method = CField(0, "uint64", default=0)
    capacity = CField(1, "int64", const=True, default=2)
    values = CField(2, "real", default=0.0, pointer=True, length="capacity")
    derivatives = CField(3, "real", default=0.0, pointer=True, length="capacity")
    z0 = CField(4, "real", default=0.0)
    dz = CField(5, "real", default=1.0)
    num_values = CField(6, "int64", default=0)

    def __init__(self, method=None, capacity=None, num_values=None, **kwargs):
        if method is not None and isinstance(method, str ):
            method_str = method.lower().strip()
            if method_str == "linear":
                method = self.INTERPOL_LINEAR
            elif method_str == "cubic":
                method = self.INTERPOL_CUBIC
            else:
                method = self.INTERPOL_NONE
            kwargs["method"]=method

        if capacity is not None or num_values is not None:
            if capacity is None:
                capacity = 1
            if num_values is None:
                num_values = capacity
            if num_values > capacity:
                capacity = num_values

            kwargs["capacity"] = capacity
            kwargs["num_values"] = num_values

        super().__init__(**kwargs)

    @staticmethod
    def ptr_offset():
        return 0

    @property
    def values_offset(self):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_values_offset( _ptr )

    @property
    def derivatives_offset(self):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_derivatives_offset( _ptr )

    def prepare_interpolation(self):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        if st_ARCH_STATUS_SUCCESS.value != \
            st_LineDensityProfileData_prepare_interpolation( _ptr ):
            raise RuntimeError( "unable to calculate derivaties for interpolation" )

    def interpol(self, z):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_value(
            _ptr, ct.c_double( z ) )

    def interpol_1st_deriv(self, z):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_1st_derivative(
            _ptr, ct.c_double( z ) )

    def interpol_2nd_deriv(self, z):
        _ptr = ct.cast( int( self._get_address() ), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_2nd_derivative(
            _ptr, ct.c_double( z ) )


def LineDensityProfileData_buffer_create_assign_address_item(
    track_job, be_sc_index, interpol_buffer_id, lp_data_index ):
    assert isinstance(track_job, TrackJob) or \
           isinstance(track_job, CudaTrackJob)
    assert interpol_buffer_id != st_ARCH_ILLEGAL_BUFFER_ID.value
    assert track_job.min_stored_buffer_id <= interpol_buffer_id
    assert track_job.max_stored_buffer_id >= interpol_buffer_id
    assert (
        track_job.stored_buffer(interpol_buffer_id).num_objects
        > lp_data_index
    )

    dest_buffer_id = st_ARCH_BEAM_ELEMENTS_BUFFER_ID.value
    src_buffer_id = interpol_buffer_id

    prev_num_assign_items = track_job.num_assign_items(
        dest_buffer_id=dest_buffer_id, src_buffer_id=src_buffer_id
    )

    _ptr_item = track_job.add_assign_address_item(
        dest_elem_type_id=st_SpaceChargeInterpolatedProfile_type_id(),
        dest_buffer_id=dest_buffer_id,
        dest_elem_index=be_sc_index,
        dest_pointer_offset= st_SpaceChargeInterpolatedProfile_interpol_data_addr_offset(
            st_NullSpaceChargeInterpolatedProfile ),
        src_elem_type_id=st_LineDensityProfileData_type_id(),
        src_buffer_id=src_buffer_id,
        src_elem_index=lp_data_index,
        src_pointer_offset=LineDensityProfileData.ptr_offset(),
    )

    return bool( prev_num_assign_items < track_job.num_assign_items(
        dest_buffer_id=dest_buffer_id, src_buffer_id=src_buffer_id ) )
