""" Manage block
"""
import os

import numpy as np

from .cobjects import CProp, CObject, CBuffer


modulepath=os.path.dirname(os.path.abspath(__file__))

try:
  import pyopencl as cl
  os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
  srcpath='-I%s'%modulepath
  src=open(os.path.join(modulepath,'block.c')).read()
  ctx = cl.create_some_context(interactive=False)
  prg=cl.Program(ctx,src).build(options=[srcpath])
  queue = cl.CommandQueue(ctx)
  mf = cl.mem_flags
  rw=mf.READ_WRITE | mf.COPY_HOST_PTR
except ImportError:
  print("Warning: error import OpenCL: track_cl not available")
  cl=False
  pass



class Drift(CObject):
    objid  = CProp('u64',0,2)
    length = CProp('f64',1,0)

class DriftExact(CObject):
    objid  = CProp('u64',0,3)
    length = CProp('f64',1,0)

class Multipole(CObject):
    objid  = CProp('u64',0,4)
    order  = CProp('f64',1,0,const=True)
    length = CProp('f64',2,0)
    hxl    = CProp('f64',3,0)
    hyl    = CProp('f64',4,0)
    bal    = CProp('f64',5,0,'2*(order+1)')
    def __init__(self,knl=[],ksl=[],**nvargs):
      if len(knl)>len(ksl):
          ksl+=[0]*(len(knl)-len(ksl))
      else:
          knl+=[0]*(len(ksl)-len(knl))
      bal=np.array(sum(zip(knl,ksl),()))
      fact=1
      for n in range(len(bal)/2):
          bal[n*2:n*2+2]/=fact
      if len(bal)>=2 and len(bal)%2==0:
          order=len(bal)/2-1
      else:
          raise ValueError("Size of bal must be even")
      nvargs['bal']=bal
      nvargs['order']=order
      CObject.__init__(self,**nvargs)




class CBlock(object):
    """ Block object
    """
    _obj_types = dict(Drift      = 2,
                      DriftExact = 3,
                      Multipole  = 4,
                      Cavity     = 5,
                      Align      = 6,
                      Block      = 7)

    def __init__(self):
        self._cbuffer=CBuffer(1)
        self.nobj=0
        self.obj={}
        self.obj_ids=[]
        self.obj_revnames={}
    def _add_obj(self,name,obj):
        self.obj[self.nobj]=obj
        self.obj.setdefault(name,[]).append(obj)
        self.obj_revnames[obj._offset]=name
        self.obj_ids.append(obj._offset)
        self.nobj+=1
    def add_Drift(self,name=None,**nvargs):
        obj=Drift(cbuffer=self._cbuffer,**nvargs)
        self._add_obj(name,obj)
    def add_Multipole(self,name=None,**nvargs):
        obj=Multipole(cbuffer=self._cbuffer,**nvargs)
        self._add_obj(name,obj)
    if cl:
        def track_cl(self,beam,nturn=1,elembyelem=None,turnbyturn=None):
            Beam=beam.__class__
            #ElemByElem data
            if elembyelem is True:
              elembyelem=Beam(npart=beam.npart*self.blocklen*nturn)
            if elembyelem is None:
              elembyelem_flag=np.uint64(0)
              elembyelem_g=cl.Buffer(ctx, rw, hostbuf=np.zeros(1))
            else:
              elembyelem_flag=np.uint64(1)
              elembyelem_g=cl.Buffer(ctx, rw, hostbuf=elembyelem._data)
            #TurnByTurn data
            if turnbyturn is True:
              self.turnbyturn=Beam(npart=beam.npart*nturn)
            if turnbyturn is None:
              turnbyturn_flag=np.uint64(0)
              turnbyturn_g=cl.Buffer(ctx, rw, hostbuf=np.zeros(1))
            else:
              turnbyturn_flag=np.uint64(1)
              turnbyturn_g=cl.Buffer(ctx, rw, hostbuf=self.turnbyturn._data)
            #Tracking data
            data_g=cl.Buffer(ctx, rw, hostbuf=self._data)
            part_g=cl.Buffer(ctx, rw, hostbuf=beam._data)
            blockid=np.uint64(self.blockid)
            nturn=np.uint64(nturn)
            npart=np.uint64(beam.npart)
            prg.Block_track(queue,[beam.npart],None,
                            data_g, part_g,
                            blockid, nturn, npart,
                            elembyelemid, turnbyturnid)
            cl.enqueue_copy(queue,self.data,data_g)
            cl.enqueue_copy(queue,beam.particles,part_g)
            if elembyelem:
                self.elembyelem=_elembyelem.get_beam()
            if turnbyturn:
                self.turnbyturn=_turnbyturn.get_beam()


