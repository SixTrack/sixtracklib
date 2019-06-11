#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import pysixtracklib as pyst
from pysixtracklib.stcommon import st_NODE_UNDEFINED_INDEX, \
    st_ARCH_STATUS_SUCCESS, st_ARCH_STATUS_GENERAL_FAILURE, \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete, \
    st_NullCudaController, st_NullCudaArgument, \
    st_Buffer_get_capacity, st_Buffer_get_size, st_Buffer_get_slot_size
import pysixtracklib_test as testlib
from pysixtracklib_test.generic_obj import GenericObj

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    num_d_values = 10
    num_e_values = 10
    num_obj      = 10

    obj_buffer = CBuffer()
    for ii in range( 0, num_obj ):
        obj = GenericObj( cbuffer=obj_buffer, type_id=ii, a=ii, b=float(ii),
            c = [ 1.0, 2.0, 3.0, 4.0 ],
            num_d=num_d_values, num_e=num_e_values )

    c_obj_buffer = st_Buffer_new_mapped_on_cbuffer( obj_buffer )
    assert c_obj_buffer != st_NullBuffer

    slot_size = st_Buffer_get_slot_size( c_obj_buffer )
    assert slot_size > 0

    c_cpy_buffer = st_Buffer_new( st_Buffer_get_capacity( c_obj_buffer ) )
    assert c_cpy_buffer != st_NullBuffer

    ctrl = pyst.CudaController()
    assert ctrl.num_nodes > 0
    assert ctrl.has_selected_node
    assert ctrl.selected_node_index != st_NODE_UNDEFINED_INDEX.value

    arg1 = pyst.CudaArgument( ctrl=ctrl )

    sys.exit(0)
