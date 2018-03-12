""" Manage block
"""
import os

import numpy as np

from .cobjects import CProp, CObject, CBuffer


modulepath = os.path.dirname(os.path.abspath(__file__))

try:
    import pyopencl as cl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    srcpath = '-I%s' % modulepath
    src = open(os.path.join(modulepath, 'block.c')).read()
    ctx = cl.create_some_context(interactive=False)
    prg = cl.Program(ctx, src).build(options=[srcpath])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    rw = mf.READ_WRITE | mf.COPY_HOST_PTR
except ImportError:
    print("Warning: error import OpenCL: track_cl not available")
    cl = False
    pass


class Drift(CObject):
    objid = CProp('u64', 0, default=2)
    length = CProp('f64', 1, default=0)


class DriftExact(CObject):
    objid = CProp('u64', 0, default=3)
    length = CProp('f64', 1, default=0)


class Multipole(CObject):
    objid = CProp('u64', 0, default=4)
    order = CProp('u64', 1, default=0, const=True)
    length = CProp('f64', 2, default=0)
    hxl = CProp('f64', 3, default=0)
    hyl = CProp('f64', 4, default=0)
    bal = CProp('f64', 5, default=0, length='2*(order+1)')

    def __init__(self, knl=[], ksl=[], **nvargs):
        if len(knl) > len(ksl):
            ksl += [0]*(len(knl)-len(ksl))
        else:
            knl += [0]*(len(ksl)-len(knl))
        bal = np.array(sum(zip(knl, ksl), ()))
        fact = 1
        for n in range(len(bal)//2):
            bal[n*2:n*2+2] /= fact
            fact *= (n+1)
        if len(bal) >= 2 and len(bal) % 2 == 0:
            order = len(bal)//2-1
        else:
            raise ValueError("Size of bal must be even")
        nvargs['bal'] = bal
        nvargs['order'] = order
        CObject.__init__(self, **nvargs)


class Cavity(CObject):
    objid = CProp('u64', 0, default=5)
    voltage = CProp('f64', 1)
    frequency = CProp('f64', 2)
    lag = CProp('f64', 3)
    lag_rad = CProp('f64', 4)

    def __init__(self, lag=0., **nvargs):
        CObject.__init__(self, lag_rad=lag/180.*np.pi, **nvargs)


class Align(CObject):
    objid = CProp('u64', 0, default=6)
    tilt = CProp('f64', 1)
    cz = CProp('f64', 2)
    sz = CProp('f64', 3)
    dx = CProp('f64', 4)
    dy = CProp('f64', 5)

    def __init__(self, tilt=0., **nvargs):
        cz = np.cos(tilt/180.*np.pi)
        sz = np.sin(tilt/180.*np.pi)
        CObject.__init__(self, cz=cz, sz=sz, **nvargs)

class Rotation(CObject):
    objid = CProp('u64', 0, default=11)
    cx  = CProp('f64', 1)
    sx  = CProp('f64', 2)
    cpx = CProp('f64', 3)
    spx = CProp('f64', 4)
    cy  = CProp('f64', 5)
    sy  = CProp('f64', 6)
    cpy = CProp('f64', 7)
    spy = CProp('f64', 8)
    ap   = CProp('f64', 9)
    h   = CProp('f64', 10)
    fRF   = CProp('f64', 11)
     
    def __init__(self,qx=0.,qy=0.,betax=0.,betay=0.,alfax=0.,alfay=0.,gamma_tr=0.,h=0.,fRF=0., **nvargs):
        gammax = (1. + alfax**2)/betax
        gammay = (1. + alfay**2)/betay
        cx  = np.cos(2.0*np.pi*qx) + alfax*np.sin(2.0*np.pi*qx)
        sx  = betax*np.sin(2.0*np.pi*qx)
        cpx = -gammax*np.sin(2.0*np.pi*qx)
        spx = np.cos(2.0*np.pi*qx) - alfax*np.sin(2.0*np.pi*qx)
        cy  = np.cos(2.0*np.pi*qy) + alfay*np.sin(2.0*np.pi*qy)
        sy  = betay*np.sin(2.0*np.pi*qy)
        cpy = -gammay*np.sin(2.0*np.pi*qy)
        spy = np.cos(2.0*np.pi*qy) - alfay*np.sin(2.0*np.pi*qy)
        ap  = 1./(gamma_tr**2)

        CObject.__init__(self, cx=cx, sx=sx, cpx=cpx, spx=spx, cy=cy, sy=sy, cpy=cpy, spy=spy, ap=ap, h=h, fRF=fRF,**nvargs)

    

class BeamBeam(CObject):
    objid = CProp('u64', 0, default=10)
    datasize = CProp('u64', 1, const=True)
    data = CProp('f64', 2, length='datasize')


    def __init__(self,
                q_part, N_part_tot, sigmaz, N_slices, min_sigma_diff, threshold_singular,
                phi, alpha, 
                Sig_11_0, Sig_12_0, Sig_13_0, 
                Sig_14_0, Sig_22_0, Sig_23_0, 
                Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0, bb_data_list, 
                **nvargs):

      import BB6D_data
      bb6d_data = BB6D_data.BB6D_init(q_part, N_part_tot, sigmaz, N_slices, min_sigma_diff, threshold_singular,
                phi, alpha, 
                Sig_11_0, Sig_12_0, Sig_13_0, 
                Sig_14_0, Sig_22_0, Sig_23_0, 
                Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0)

      bb_data_list.append(bb6d_data)
      buffer = bb6d_data.tobuffer()
      CObject.__init__(self, data=buffer, datasize=len(buffer), **nvargs)

class CBlock(object):
    """ Block object
    """
    _elem_types = dict(Drift=2,
                       DriftExact=3,
                       Multipole=4,
                       Cavity=5,
                       Align=6,
                       Block=7,
                       BeamBeam=10,
                       Rotation=11)

    bb_data_list = []

    def __init__(self):
        self._cbuffer = CBuffer(1)
        self.nelems = 0
        self.elem = {}
        self.elem_ids = []
        self.elem_revnames = {}

    def _add_elem(self, name, elem):
        self.elem[self.nelems] = elem
        self.elem.setdefault(name, []).append(elem)
        self.elem_revnames[elem._offset] = name
        self.elem_ids.append(elem._offset)
        self.nelems += 1

    def add_Drift(self, name=None, **nvargs):
        elem = Drift(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)

    def add_Multipole(self, name=None, **nvargs):
        elem = Multipole(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)

    def add_Cavity(self, name=None, **nvargs):
        elem = Cavity(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)

    def add_Align(self, name=None, **nvargs):
        elem = Align(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)

    def add_BeamBeam(self, name=None, **nvargs):
        elem = BeamBeam(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)

    def add_BeamBeam(self,name=None,**nvargs):
        elem=BeamBeam(cbuffer=self._cbuffer, bb_data_list = self.bb_data_list, **nvargs)
        self._add_elem(name,elem)

    def add_Rotation(self, name=None, **nvargs):
        elem = Rotation(cbuffer=self._cbuffer, **nvargs)
        self._add_elem(name, elem)
    if cl:
        def track_cl(self, particles, nturns=1, elembyelem=None, turnbyturn=None):
            CParticles = particles.__class__
            npart = np.int64(particles.npart)
            # uint bug in boost/pyopencl/numpy???
            particles_g = cl.Buffer(ctx, rw, hostbuf=particles._cbuffer.data)
            # ElemByElem data
            if elembyelem is True:
                elembyelem = CParticles(npart=npart*self.nelems*nturns)
                elembyelem = elembyelem.reshape(nturns, self.nelems, npart)
            if elembyelem is None:
                elembyelem_g = cl.Buffer(ctx, rw, hostbuf=np.array([0]))
            else:
                elembyelem_g = cl.Buffer(
                    ctx, rw, hostbuf=elembyelem._cbuffer.data)
            # TurnByTurn data
            if turnbyturn is True:
                turnbyturn = CParticles(npart=npart*(nturns+1))
                turnbyturn = turnbyturn.reshape(nturns+1, npart)
            if turnbyturn is None:
                turnbyturn_g = cl.Buffer(ctx, rw, hostbuf=np.array([0]))
            else:
                turnbyturn_g = cl.Buffer(
                    ctx, rw, hostbuf=turnbyturn._cbuffer.data)
            # Tracking data
            elems_g = cl.Buffer(ctx, rw, hostbuf=self._cbuffer.data)
            elemids = np.array(self.elem_ids, dtype='uint64')
            elemids_g = cl.Buffer(ctx, rw, hostbuf=elemids)
            nelems = np.int64(self.nelems)
            nturns = np.int64(nturns)
            prg.Block_unpack(queue, [1], None,
                             particles_g, elembyelem_g, turnbyturn_g)
            prg.Block_track(queue, [npart], None,
                            elems_g, elemids_g, nelems,
                            nturns,
                            particles_g, elembyelem_g, turnbyturn_g)
            cl.enqueue_copy(queue, particles._cbuffer.data, particles_g)
            if turnbyturn:
                cl.enqueue_copy(queue, turnbyturn._cbuffer.data, turnbyturn_g)
            if elembyelem:
                cl.enqueue_copy(queue, elembyelem._cbuffer.data, elembyelem_g)
            return particles, elembyelem, turnbyturn
