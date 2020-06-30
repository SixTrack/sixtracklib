#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import ctypes as ct
from cobjects import CBuffer, CObject, CField
from .stcommon import st_ARCH_STATUS_SUCCESS, st_LineDensityProfileData_p, \
    st_LineDensityProfileData_type_id, \
    st_NullLineDensityProfileData, \
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

    method = CField(0, "uint64", default=INTERPOL_LINEAR, alignment=8)
    num_values = CField(1, "int64", default=0, alignment=8)
    values = CField(2, "real", default=0.0, pointer=True,
                    length="capacity", alignment=8)
    derivatives = CField(3, "real", default=0.0, pointer=True,
                    length="capacity", alignment=8)
    z0 = CField(4, "real", default=0.0, alignment=8)
    dz = CField(5, "real", default=1.0, alignment=8)
    capacity = CField(6, "int64", const=True, alignment=8, default=1)

    def __init__(self, **kwargs):
        if "method" in kwargs and isinstance(kwargs["method"], str ):
            method_str = kwargs["method"].lower().strip()
            if method_str == "linear":
                kwargs["method"] = self.INTERPOL_LINEAR
            elif method_str == "cubic":
                kwargs["method"] = self.INTERPOL_CUBIC
            else:
                kwargs["method"] = self.INTERPOL_NONE

        if "values" in kwargs and kwargs[ "values" ] is not None:
            try:
                temp_num_values = len( kwargs["values"] )
            except TypeError:
                temp_num_values = 0
            if "num_values" not in kwargs or \
                temp_num_values > kwargs[ "num_values" ]:
                kwargs[ "num_values" ] = temp_num_values

        if "num_values" in kwargs and "capacity" in kwargs and \
            kwargs[ "num_values" ] > kwargs[ "capacity" ]:
            kwargs[ "capacity" ] = kwargs[ "num_values" ]
        elif "capacity" in kwargs and "num_values" in kwargs and \
            kwargs[ "capacity" ] > 0 and kwargs[ "num_values" ] == 0:
            kwargs[ "num_values" ] = kwargs[ "capacity" ]

        if "capacity" in kwargs and kwargs[ "capacity" ] and \
            "method" in kwargs and kwargs["method"] == self.INTERPOL_NONE and \
            kwargs[ "capacity" ] < 2:
            raise ValueError( "linear interpol. requires >= 2 values" )
        elif "capacity" in kwargs and kwargs[ "capacity" ] and \
            "method" in kwargs and kwargs["method"] == self.INTERPOL_CUBIC \
            and kwargs[ "capacity" ] < 4:
            raise ValueError( "cubic spline interpol. requires >= 4 values" )

        if "dz" in kwargs and kwargs["dz"] <= 0.0:
            raise ValueError( "dz is required to be > 0" )
        print( kwargs )
        super().__init__(**kwargs)

    @property
    def values_offset(self):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_values_offset( _ptr )

    @property
    def derivatives_offset(self):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_derivatives_offset( _ptr )

    def prepare_interpolation(self):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        if st_ARCH_STATUS_SUCCESS.value != \
            st_LineDensityProfileData_prepare_interpolation( _ptr ):
            raise RuntimeError( "unable to calculate derivaties for interpolation" )

    def interpol(self, z):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_value(
            _ptr, ct.c_double( z ) )

    def interpol_1st_deriv(self, z):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_1st_derivative(
            _ptr, ct.c_double( z ) )

    def interpol_2nd_deriv(self, z):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_interpolate_2nd_derivative(
            _ptr, ct.c_double( z ) )
