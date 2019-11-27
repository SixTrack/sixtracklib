#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from cobjects import CBuffer, CObject, CField

from .trackjob import TrackJob
from .buffer import AssignAddressItem, get_cbuffer_from_obj, Buffer
from .stcommon import st_ARCH_ILLEGAL_BUFFER_ID, st_ARCH_BEAM_ELEMENTS_BUFFER_ID

st_OBJECT_TYPE_TRICUB = 32768
st_OBJECT_TYPE_TRICUB_DATA = 32769
st_TRICUB_PTR_OFFSET = 32
st_TRICUB_DATA_PTR_OFFSET = 0

def TriCub_buffer_create_assign_address_item(
    track_job, be_tricub_index, tricub_data_buffer_id, tricub_data_index ):
    assert isinstance( track_job, TrackJob )
    assert tricub_data_buffer_id != st_ARCH_ILLEGAL_BUFFER_ID.value
    assert track_job.min_stored_buffer_id <= tricub_data_buffer_id
    assert track_job.max_stored_buffer_id >= tricub_data_buffer_id
    assert track_job.stored_buffer( tricub_data_buffer_id ).num_objects > \
           tricub_data_index

    dest_buffer_id = st_ARCH_BEAM_ELEMENTS_BUFFER_ID.value
    src_buffer_id = tricub_data_buffer_id

    prev_num_assign_items = track_job.num_assign_items(
        dest_buffer_id=dest_buffer_id, src_buffer_id=src_buffer_id)

    _ptr_item = track_job.add_assign_address_item(
        dest_elem_type_id=st_OBJECT_TYPE_TRICUB,
        dest_buffer_id=tricub_data_buffer_id,
        dest_elem_index=be_tricub_index,
        dest_pointer_offset=st_TRICUB_PTR_OFFSET,
        src_elem_type_id=st_OBJECT_TYPE_TRICUB_DATA,
        src_buffer_id=tricub_data_buffer_id,
        src_elem_index=tricub_data_index,
        src_pointer_offset=st_TRICUB_DATA_PTR_OFFSET)

    if prev_num_assign_items < track_job.num_assign_items(
        dest_buffer_id=dest_buffer_id, src_buffer_id=src_buffer_id ):
        return True

    return False



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
    mirror_x = CField(9, 'int64',  default=0.0, alignment=8)
    mirror_y = CField(10, 'int64',  default=0.0, alignment=8)
    mirror_z = CField(11, 'int64',  default=0.0, alignment=8)
    table_addr = CField(12, 'uint64', default=0, pointer=True,
                    length='nx * ny * nz * 8', alignment=8 )

    def __init__(self, nx=0, ny=0, nz=0, **kwargs):
        super().__init__(nx=nx, ny=ny, nz=nz, **kwargs)


class TriCub(CObject):
    _typeid = st_OBJECT_TYPE_TRICUB
    x = CField(0, 'real', default=0.0, alignment=8)
    y = CField(1, 'real', default=0.0, alignment=8)
    z = CField(2, 'real', default=0.0, alignment=8)
    length = CField(3, 'real', default=0.0, alignment=8)
    data_addr = CField(4, 'uint64', default=0, alignment=8)

