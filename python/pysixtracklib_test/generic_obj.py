#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CObject, CField
import numpy as np


class GenericObj(CObject):
    _typeid = 99999
    type_id = CField(0, 'int64', default=_typeid, alignment=8)
    a = CField(1, 'int32', default=0, alignment=8)
    b = CField(2, 'real', default=0.0, alignment=8)
    c = CField(3, 'real', length=4, default=0.0, alignment=8)
    num_d = CField(4, 'uint64', const=True, default=0, alignment=8)
    d = CField(5, 'uint8', default=0, pointer=True,
               length='num_d', alignment=8)
    num_e = CField(6, 'uint64', const=True, default=0, alignment=8)
    e = CField(7, 'real', default=0.0, pointer=True,
               length='num_e', alignment=8)

    def __init__(self, num_d=0, num_e=0, d=None, e=None, **kwargs):
        in_d_len = d is not None and d and len(d) or 0
        in_e_len = e is not None and e and len(e) or 0

        d_len = max(num_d, in_d_len)

        if d is None or not(d and len(d) > 0):
            d = np.zeros(d_len, dtype=np.dtype('uint8'))
        elif d and len(d) > 0:
            _d = np.zeros(d_len, dtype=np.dtype('uint8'))
            _d[:len(d)] = d
            d = _d

        if d is not None and len(d) == d_len:
            kwargs["d"] = d

        e_len = max(num_e, in_e_len)

        if e is None or not(e and len(e) > 0):
            e = np.zeros(e_len, dtype=np.dtype('float64'))
        elif e and len(e) > 0:
            _e = np.zeros(e_len, dtype=np.dtype('float64'))
            _e[:len(e)] = e
            e = _e

        if e is not None and len(e) == e_len:
            kwargs["e"] = e

        kwargs["num_d"] = d_len
        kwargs["num_e"] = e_len

        CObject.__init__( self,**kwargs)


# end: python/pysixtracklib_test/generic_obj.py
