#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CBuffer, CObject, CField
import numpy as np


class Drift(CObject):
    _typeid = 2
    length = CField(0, 'real', default=0.0, alignment=8)


class DriftExact(CObject):
    _typeid = 3
    length = CField(0, 'real', default=0.0, alignment=8)


class BeamMonitor(CObject):
    _typeid = 10
    num_stores      = CField( 0, 'int64',  default=0, alignment=8 )
    start           = CField( 1, 'int64',  default=0, alignment=8 )
    skip            = CField( 2, 'int64',  default=1, alignment=8 )
    out_address     = CField( 3, 'uint64', default=0, alignment=8 )
    max_particle_id = CField( 4, 'int64',  default=0, alignment=8 )
    min_particle_id = CField( 5, 'int64',  default=0, alignment=8 )
    is_rolling      = CField( 6, 'int64',  default=0, alignment=8 )
    is_turn_ordered = CField( 7, 'int64',  default=1, alignment=8 )


class Multipole(CObject):
    _typeid = 4
    order = CField(0, 'int64',   default=0,    alignment=8)
    length = CField(1, 'real',    default=0.0,  alignment=8)
    hxl = CField(2, 'real',    default=0.0,  alignment=8)
    hyl = CField(3, 'real',    default=0.0,  alignment=8)
    bal = CField(4, 'real',    default=0.0,
                 length='2 * order + 2', pointer=True, alignment=8)

    def _factorial(self, x):
        if not isinstance(x, int):
            return 0
        return (x > 0) and (x * self._factorial(x - 1)) or 1

    def __init__(self, order=None, knl=None, ksl=None, bal=None, **kwargs):

        if bal is None and (not(knl is None) or not(ksl is None)):
            if knl is None:
                knl = []
            if ksl is None:
                ksl = []
            if order is None:
                order = 0

            n = max((order + 1), max(len(knl), len(ksl)))
            _knl = np.array(knl)
            nknl = np.zeros(n, dtype=_knl.dtype)
            nknl[:len(knl)] = knl
            knl = nknl
            del(_knl)

            _ksl = np.array(ksl)
            nksl = np.zeros(n, dtype=_ksl.dtype)
            nksl[:len(ksl)] = ksl
            ksl = nksl
            del(_ksl)

            assert(n > 0)
            order = n - 1

            bal = np.zeros(2 * order + 2)
            assert(len(knl) == len(ksl))

            for ii in range(0, len(knl)):
                inv_factorial = 1.0 / float(self._factorial(ii))
                jj = 2 * ii
                bal[jj] = knl[ii] * inv_factorial
                bal[jj + 1] = ksl[ii] * inv_factorial

        elif not(bal is None) and bal and \
                len(bal) > 2 and ((len(bal) % 2) == 0):

            order = (len(bal) - 2) / 2
            assert(order > 0)

        elif bal is None and knl is None and ksl is None and \
                not(order is None) and order > 0:
            bal = np.zeros(2 * order + 2)

        if not(bal is None or order is None):
            CObject.__init__(self, bal=bal, order=order, **kwargs)
        else:
            CObject.__init__(self, bal=[], order=0, **kwargs)


class Cavity(CObject):
    _typeid = 5
    voltage = CField(0, 'real', default=0.0,  alignment=8)
    frequency = CField(1, 'real', default=0.0,  alignment=8)
    lag = CField(2, 'real', default=0.0,  alignment=8)


class XYShift(CObject):
    _typeid = 6
    dx = CField(0, 'real',   default=0.0,  alignment=8)
    dy = CField(1, 'real',   default=0.0,  alignment=8)


class SRotation(CObject):
    _typeid = 7
    cos_z = CField(0, 'real',   default=1.0,  alignment=8)
    sin_z = CField(1, 'real',   default=0.0,  alignment=8)

    def __init__(self, angle=0, **nargs):
        anglerad = angle/180*np.pi
        cos_z = np.cos(anglerad)
        sin_z = np.sin(anglerad)
        CObject.__init__(self,
                         cos_z=cos_z, sin_z=sin_z, **nargs)


class BeamBeam4D(CObject):
    _typeid = 8
    size = CField(0, 'uint64', const=True, default=0)
    data = CField(1, 'float64',   default=0.0,
                  length='size', pointer=True)

    def __init__(self, data=None, **kwargs):
        if data is None:
            slots = ('q_part', 'N_part', 'sigma_x', 'sigma_y', 'beta_s',
                     'min_sigma_diff', 'Delta_x', 'Delta_y', 'Dpx_sub', 'Dpy_sub', 'enabled')
            data = [kwargs[ss] for ss in slots]
            CObject.__init__(self, size=len(data), data=data, **kwargs)
        else:
            CObject.__init__(self, **kwargs)


class BeamBeam6D(CObject):
    _typeid = 9
    size = CField(0, 'uint64', const=True, default=0)
    data = CField(1, 'float64',   default=0.0,
                  length='size', pointer=True)

    def __init__(self, data=None, **kwargs):
        if data is None:
            import pysixtrack
            data = pysixtrack.BB6Ddata.BB6D_init(
                **{kk: kwargs[kk] for kk in kwargs.keys() if kk != 'cbuffer'}).tobuffer()
            CObject.__init__(self, size=len(data), data=data, **kwargs)
        else:
            CObject.__init__(self, **kwargs)


class Elements(object):
    element_types = {'Cavity': Cavity,
                     'Drift': Drift,
                     'DriftExact': DriftExact,
                     'Multipole': Multipole,
                     #                     'RFMultipole': RFMultipole,
                     'SRotation': SRotation,
                     'XYShift': XYShift,
                     'BeamBeam6D': BeamBeam6D,
                     'BeamBeam4D': BeamBeam4D,
                     #                     'Line': Line,
                     'BeamMonitor': BeamMonitor,
                     }

    def _mk_fun(self, buff, cls):
        def fun(*args, **nargs):
            # print(cls.__name__,nargs)
            return cls(cbuffer=buff, **nargs)
        return fun

    @classmethod
    def fromfile(cls, filename):
        cbuffer = CBuffer.fromfile(filename)
        return cls(cbuffer=cbuffer)

    @classmethod
    def fromline(cls, line):
        self = cls()
        for label, element_name, element in line:
            getattr(self, element_name)(**element._asdict())
        return self

    def tofile(self, filename):
        self.cbuffer.tofile(filename)

    def __init__(self, cbuffer=None):
        if cbuffer is None:
            self.cbuffer = CBuffer()
        else:
            self.cbuffer = cbuffer
        for name, cls in self.element_types.items():
            setattr(self, name, self._mk_fun(self.cbuffer, cls))
            self.cbuffer.typeids[cls._typeid] = cls

    def gen_builder(self):
        out = {}
        for name, cls in self.element_types.items():
            out[name] = getattr(self, name)
        return out

    def get_elements(self):
        n = self.cbuffer.n_objects
        return [self.cbuffer.get_object(i) for i in range(n)]

    def get(self, objid):
        return self.cbuffer.get_object(objid)
