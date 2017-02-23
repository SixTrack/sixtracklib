import ctypes

import numpy as np

particle_t=np.dtype([
         ('partid','int32'),
         ('elemid','int32'),
         ('turn'  ,'int32'),
         ('state' ,'int32'),
         ('s'     ,float),
         ('x'     ,float),
         ('px'    ,float),
         ('y'     ,float),
         ('py'    ,float),
         ('sigma' ,float),
         ('psigma',float),
         ('chi'   ,float),
         ('delta' ,float),
         ('rpp'   ,float),
         ('rvv'   ,float),
         ('beta'  ,float),
         ('gamma' ,float),
         ('m0'    ,float),
         ('q0'    ,float),
         ('q'     ,float),
         ('beta0' ,float),
         ('gamma0',float),
         ('p0c',float),
         ])

class cBeam_ctypes(ctypes.Structure):
      _fields_ = [("npart",     ctypes.c_uint64),
                  ("particles", ctypes.c_void_p)]


class cBeam(object):
  clight=299792458
  pi=3.141592653589793238
  pcharge=1.602176565e-19
  echarge=-pcharge
  emass=0.510998928e6
  pmass=938.272046e6
  epsilon0=8.854187817e-12
  mu0=4e-7*pi
  eradius=pcharge**2/(4*pi*epsilon0*emass*clight**2)
  pradius=pcharge**2/(4*pi*epsilon0*pmass*clight**2)
  anumber=6.022140857e23
  kboltz=8.6173303e-5 #ev K^-1 #1.38064852e-23 #   JK^-1
  def __init__(self,npart=None,m0=pmass,p0c=450,q0=pcharge,particles=None):
    if particles is None:
      self.npart=npart
      self.particles=np.zeros(npart,particle_t)
      self.particles['m0']=m0
      e0=np.sqrt(p0c**2+m0**2)
      gamma0=e0/m0
      beta0=p0c/m0/gamma0
      chi=1.
      self.particles['partid']=np.arange(npart)
      self.particles['chi']=chi
      self.particles['beta0']=beta0
      self.particles['gamma0']=gamma0
      self.particles['p0c']=p0c
      self.particles['rvv']=1.
      self.particles['rpp']=1.
    else:
      self.particles=particles.view(particle_t)
      self.npart=len(self.particles)
  def ctypes(self):
    cdata=cBeam_ctypes(self.npart,self.particles.ctypes.data)
    return ctypes.pointer(cdata)
  def get_size(self):
    return self.npart*particle_t.itemsize/8


