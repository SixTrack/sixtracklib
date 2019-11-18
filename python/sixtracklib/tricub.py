#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from cobjects import CBuffer, CObject, CField
import ctypes as ct

from .stcommon import sixtracklib
from .stcommon import st_Buffer_p, st_NullBuffer, st_buffer_size_t, \
      st_AssignAddressItem_p, st_NullAssignAddressItem

from .buffer import AssignAddressItem

st_OBJECT_TYPE_TRICUB = 32768
st_OBJECT_TYPE_TRICUB_DATA = 32769

st_TriCub_buffer_create_assign_address_item = \
    sixtracklib.st_TriCub_buffer_create_assign_address_item
st_TriCub_buffer_create_assign_address_item.argtypes = [ st_Buffer_p,
    st_Buffer_p, st_buffer_size_t, st_Buffer_p, st_buffer_size_t ]
st_TriCub_buffer_create_assign_address_item.restype = st_AssignAddressItem_p

def TriCub_buffer_create_assign_address_item(map_buffer,
    beam_elements, belements_index, tricub_data_buffer, tricub_data_index):
    item = AssignAddressItem(cbuffer=map_buffer,
        dest_elem_type_id=st_OBJECT_TYPE_TRICUB,
        dest_buffer_id=1,
        dest_elem_index=belements_index,
        dest_pointer_offset=32,
        src_elem_type_id=st_OBJECT_TYPE_TRICUB_DATA,
        src_buffer_id=4,
        src_elem_index=tricub_data_index,
        src_pointer_offset=0 )
    return item


class TriCubData(CObject):
    _typeid = st_OBJECT_TYPE_TRICUB_DATA
    x0 = CField(0, 'real',  default=0.0, alignment=8)
    dx = CField(1, 'real',  default=0.0, alignment=8)
    nx = CField(2, 'int64', default=0,   alignment=8, const=True)
    y0 = CField(3, 'real',  default=0.0, alignment=8)
    dy = CField(4, 'real',  default=0.0, alignment=8)
    ny = CField(5, 'int64', default=0,   alignment=8, const=True)
    z0 = CField(6, 'real',  default=0.0, alignment=8)
    dz = CField(7, 'real',  default=0.0, alignment=8)
    nz = CField(8, 'int64', default=0,   alignment=8, const=True)
    table_addr = CField(9, 'uint64', default=0, pointer=True,
                           length='nx * ny * nz * 8', alignment=8 )

    def __init__(self, nx=0, ny=0, nz=0, **kwargs ):
        super().__init__( nx=nx, ny=ny, nz=nz, **kwargs )


class TriCub(CObject):
    _typeid = st_OBJECT_TYPE_TRICUB
    x = CField(0, 'real', default=0.0, alignment=8)
    y = CField(1, 'real', default=0.0, alignment=8)
    z = CField(2, 'real', default=0.0, alignment=8)
    length = CField(3, 'real', default=0.0, alignment=8)
    data_addr = CField(4, 'uint64', default=0, alignment=8)



