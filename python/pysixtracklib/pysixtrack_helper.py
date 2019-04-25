import importlib
from importlib import util

if importlib.util.find_spec('pysixtrack') is not None:
    from pysixtrack import track as pysixelem
    import numpy as np
    from cobjects import CBuffer, CObject, CField
    from .beam_elements import \
        Drift, DriftExact, Multipole, Cavity, XYShift, SRotation, \
        BeamBeam4D, BeamMonitor

    class BeamElementConverter(object):
        _type_id_to_elem_map = {
            2: Drift, 3: DriftExact, 4: Multipole, 5: Cavity, 6: XYShift,
            7: SRotation, 8: BeamBeam4D,  # 9:BeamBeam6D,
            10: BeamMonitor}

        _pysixtrack_to_type_id_map = {
            'Drift': 2, 'DriftExact': 3, 'Multipole': 4, 'Cavity': 5,
            'XYShift': 6, 'SRotation': 7, 'BeamBeam4D': 8,  # 'BeamBeam6D': 9,
            'BeamMonitor': 10, }

        @staticmethod
        def get_elemtype(type_id):
            return BeamElementConverter._type_id_to_elem_map.get(type_id, None)

        @classmethod
        def get_typeid(cls):
            inv_map = {v: k for k, v in
                       BeamElementConverter._type_id_to_elem_map.items()}
            return inv_map.get(cls, None)

        @staticmethod
        def to_pysixtrack(elem):
            pysixtrack_elem = None

            try:
                type_id = elem._typeid
            except AttributeError:
                type_id = None

            cls = type_id is not None and \
                BeamElementConverter._type_id_to_elem_map.get(
                    type_id, None) or None

            if cls is not None:
                if type_id == 2:
                    pysixtrack_elem = pysixelem.Drift(length=elem.length)
                elif type_id == 3:
                    pysixtrack_elem = pysixelem.DriftExact(length=elem.length)
                elif type_id == 4:
                    pysixtrack_elem = pysixelem.Multipole(
                        knl=elem.knl, ksl=elem.ksl,
                        hxl=elem.hxl, hyl=elem.hyl, length=elem.length)
                elif type_id == 5:
                    pysixtrack_elem = pysixelem.Cavity(
                        voltage=elem.voltage, frequency=elem.frequency,
                        lag=elem.lag)
                elif type_id == 6:
                    pysixtrack_elem = pysixelem.XYShift(dx=elem.dx, dy=elem.dy)
                elif type_id == 7:
                    pysixtrack_elem = pysixelem.SRotation(angle=elem.angle_deg)
                elif type_id == 8:
                    pysixtrack_elem = pysixelem.BeamBeam4D(
                        q_part=elem.q_part, N_part=elem.N_part,
                        sigma_x=elem.sigma_x, sigma_y=elem.sigma_y,
                        beta_s=elem.beta_s,
                        min_sigma_diff=elem.min_sigma_diff,
                        Delta_x=elem.Delta_x, Delta_y=elem.Delta_y,
                        Dpx_sub=elem.Dpx_sub, Dpy_sub=elem.Dpy_sub,
                        enabled=elem.enabled)
                # elif type_id == 9:
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
        def from_pysixtrack(elem, cbuffer=None):
            elem_type = elem.__class__.__name__

            if elem_type in BeamElementConverter._pysixtrack_to_type_id_map:
                type_id = BeamElementConverter._pysixtrack_to_type_id_map[
                    elem_type]
                if type_id in BeamElementConverter._type_id_to_elem_map:
                    cls = BeamElementConverter._type_id_to_elem_map[type_id]
                    return cls(cbuffer=cbuffer, **elem.as_dict())

            return None
else:
    class BeamElementConverter(object):
        pass
