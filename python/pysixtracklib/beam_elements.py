#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CBuffer, CObject, CField
import itertools
import numpy as np
from pysixtrack import track as pysixelem

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
    order  = CField(0, 'int64', default=0, const=True, alignment=8)
    length = CField(1, 'real',  default=0.0,  alignment=8)
    hxl    = CField(2, 'real',  default=0.0,  alignment=8)
    hyl    = CField(3, 'real',  default=0.0,  alignment=8)
    bal    = CField(4, 'real',  default=0.0,  alignment=8, pointer=True,
                                length='2 * order + 2' )

    @staticmethod
    def _factorial( x ):
        if not isinstance(x, int):
            return 0
        return (x > 0) and (x * Multipole._factorial(x - 1)) or 1

    def __init__(self, order=None, knl=None, ksl=None, bal=None, **kwargs):
        if bal is None and \
            (knl is not None or ksl is not None or order is not None):
            if knl is None:
                knl = []
            if ksl is None:
                ksl = []
            if order is None:
                order = 0

            n = max((order + 1), max(len(knl), len(ksl)))
            assert(n > 0)

            _knl = np.array(knl)
            nknl = np.zeros(n, dtype=_knl.dtype)
            nknl[:len(knl)] = knl
            knl = nknl
            del(_knl)
            assert(len(knl) == n )

            _ksl = np.array(ksl)
            nksl = np.zeros(n, dtype=_ksl.dtype)
            nksl[:len(ksl)] = ksl
            ksl = nksl
            del(_ksl)
            assert(len(ksl) == n)

            order = n-1
            bal = np.zeros(2 * order + 2)

            for ii in range(0, len(knl)):
                inv_factorial = 1.0 / float(Multipole._factorial(ii))
                jj = 2 * ii
                bal[jj] = knl[ii] * inv_factorial
                bal[jj + 1] = ksl[ii] * inv_factorial

            kwargs[ "bal" ] = bal
            kwargs[ "order" ] = order

        elif bal is not None and bal and len(bal) > 2 and ((len(bal) % 2) == 0):
            kwargs[ "bal" ] = bal
            kwargs[ "order" ] = (len(bal) - 2) / 2

        CObject.__init__( self, **kwargs )


    @property
    def knl( self ):
        return [ self.bal[ ii ] * Multipole._factorial( int( ii / 2 ) )
                  for ii in range( 0, len( self.bal ), 2 ) ]

    @property
    def ksl( self ):
        return [ self.bal[ ii + 1 ] * Multipole._factorial( int( ii / 2 ) + 1 )
                    for ii in range( 0, len( self.bal ), 2 ) ]



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

    @property
    def angle( self ):
        return np.arctan2( self.sin_z, self.cos_z )

    @property
    def angle_deg( self ):
        return self.angle * ( 180.0 / np.pi )


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


class BeamElement(object):
    _type_id_to_elem_map = {
        2: Drift, 3: DriftExact, 4:Multipole, 5:Cavity, 6:XYShift, 7:SRotation,
        #9:BeamBeam6D,
        8:BeamBeam4D, 10:BeamMonitor }

    _pysixtrack_to_type_id_map = {
        'Drift': 2, 'DriftExact': 3, 'Multipole': 4, 'Cavity': 5, 'XYShift': 6,
        'SRotation': 7, 'BeamBeam4D': 8, #'BeamBeam6D': 9,
        'BeamMonitor': 10, }

    @staticmethod
    def get_elemtype( type_id ):
        return BeamElement._type_id_to_elem_map.get( type_id, None )

    @classmethod
    def get_typeid( cls ):
        inv_map = { v: k for k, v in BeamElement._type_id_to_elem_map.items() }
        return inv_map.get( cls, None )

    @staticmethod
    def to_pysixtrack( elem ):
        pysixtrack_elem = None

        try:
            type_id = elem._typeid
        except AttributeError:
            type_id = None

        if type_id is None or \
            BeamElement._type_id_to_elem_map.get( type_id, None ) is None:

            if type_id == 2:
                pysixtrack_elem = pysixelem.Drift( length=elem.length )
            elif type_id == 3:
                pysixtrack_elem = pysixelem.DriftExact( length=elem.length )
            elif type_id == 4:
                pysixtrack_elem = pysixelem.Multipole(
                    knl=elem.knl, ksl=elem.ksl,
                    hxl=elem.hxl, hyl=elem.hyl, length=elem.length )
            elif type_id == 5:
                pysixtrack_elem = pysixelem.Cavity( voltage=elem.voltage,
                    frequency=elem.frequency, lag=elem.lag )
            elif type_id == 6:
                pysixtrack_elem = pysixelem.XYShift( dx=elem.dx, dy=elem.dy )
            elif type_id == 7:
                pysixtrack_elem = pysixelem.SRotation( angle=elem.angle_deg )
            elif type_id == 8:
                pysixtrack_elem = pysixelem.BeamBeam4D(
                    q_part=elem.q_part, N_part=elem.N_part,
                    sigma_x=elem.sigma_x, sigma_y=elem.sigma_y,
                    beta_s=elem.beta_s,
                    min_sigma_diff=elem.min_sigma_diff,
                    Delta_x=elem.Delta_x, Delta_y=elem.Delta_y,
                    Dpx_sub=elem.Dpx_sub, Dpy_sub=elem.Dpy_sub,
                    enabled=elem.enabled)
            #elif type_id == 9:
                #pysixtrack_elem = pysixelem.BeamBeam6D()
            elif type_id == 10:
                pysixtrack_elem = pysixelem.BeamMonitor(
                    num_stores=elem.num_stores, start=elem.start,
                    skip=elem.skip, max_particle_id=elem.max_particle_id,
                    min_particle_id=elem.min_particle_id,
                    is_rolling=elem.is_rolling,
                    is_turn_ordered=elem.is_turn_ordered,)

        return pysixtrack_elem

    @staticmethod
    def from_pysixtrack( elem, cbuffer=None ):
        elem_type = elem.__class__.__name__

        if elem_type in BeamElement._pysixtrack_to_type_id_map:
            type_id = BeamElement._pysixtrack_to_type_id_map[ elem_type ]
            if type_id in BeamElement._type_id_to_elem_map:
                cls = BeamElement._type_id_to_elem_map[ type_id ]
                return cls( cbuffer=cbuffer, **elem.as_dict() )

        return None



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
