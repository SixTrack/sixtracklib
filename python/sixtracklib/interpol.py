#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import ctypes as ct
from cobjects import CBuffer, CObject, CField
from .stcommon import st_LineDensityProfileData_p, \
    st_LineDensityProfileData_type_id, \
    st_NullLineDensityProfileData, \
    st_LineDensityProfileData_values_offset, \
    st_LineDensityProfileData_derivatives_offset, \
    st_LineDensityProfileData_prepare_interpolation

class LineDensityProfileData(CObject):
    _typeid = st_LineDensityProfileData_type_id()

    INTERPOL_LINEAR = 0
    INTERPOL_CUBIC = 1
    INTERPOL_NONE = 255

    method = CField(0, "uint64", default=INTERPOL_LINEAR, alignment=8)
    num_values = CField(1, "int64", default=0, alignment=8)
    values = CField(2, "real", default=0.0, pointer=True,
                    length="capacity", alignment=8)
    derivatives = CField(3, "real", pointer=True,
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

        if not "method" in kwargs:
            kwargs["method"] = self.INTERPOL_NONE

        assert "method" in kwargs

        if kwargs["method"] == self.INTERPOL_NONE and num_values < 2:
            raise ValueError( "linear interpol. requires >= 2 values" )

        elif kwargs["method"] == self.INTERPOL_CUBIC and num_values < 4:
            raise ValueError( "cubic spline interpol. requires >= 4 values" )

        if "dz" in kwargs and kwargs["dz"] <= 0.0:
            raise ValueError( "dz is required to be positive and > 0" )

        super().__init__(**kwargs)

    @property
    def values_offset(self):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_values_offset( _ptr )

    @property
    def derivatives_offset(self):
        _ptr = ct.cast( self._get_address(), st_LineDensityProfileData_p )
        return st_LineDensityProfileData_derivatives_offset( _ptr )



