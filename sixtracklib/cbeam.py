import ctypes

import numpy as np

particle_t=np.dtype([
         ('partid' ,'int32'),
         ('elemid' ,'int32'),
         ('turn'   ,'int32'),
         ('state'  ,'int32'),
         ('s'      ,float),
         ('x'      ,float),
         ('px'     ,float),
         ('y'      ,float),
         ('py'     ,float),
         ('sigma'  ,float),
         ('psigma' ,float),
         ('chi'    ,float),
         ('delta'  ,float),
         ('rpp'    ,float),
         ('rvv'    ,float),
         ('beta'   ,float),
         ('gamma'  ,float),
         ('mass0'  ,float),
         ('charge0',float),
         ('charge' ,float),
         ('beta0'  ,float),
         ('gamma0' ,float),
         ('p0c'    ,float),
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
  @classmethod
  def from_full_beam(cls,beam):
      npart=len(beam['x'])
      particles=np.zeros(npart,particle_t)
      for nn in particle_t.names:
         particles[nn]=beam[nn]
      return cls(particles=particles)
  pt =property(lambda p: (p.psigma*p.beta0))
  pc =property(lambda p: (p.beta*p.gamma*p.mass0))
  energy =property(lambda p: (p.gamma*p.mass0))
  def __init__(self,npart=None,mass0=pmass,p0c=450,q0=1.0,particles=None):
    if particles is None:
      self.npart=npart
      self.particles=np.zeros(npart,particle_t)
      self.particles['mass0']=mass0
      energy0=np.sqrt(p0c**2+mass0**2)
      gamma0=energy0/mass0
      beta0=p0c/mass0/gamma0
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
  def copy(self):
    return self.__class__(particles=self.particles.copy())
  def __getitem__(self,kk):
    particles=self.particles.copy().__getitem__(kk)
    return self.__class__(particles=particles)
  def get_size(self):
    return self.npart*particle_t.itemsize/8
  def __getattr__(self,kk):
    return self.particles[kk]
  def __dir__(self):
    return sorted(particle_t.names)
  def compare(self,ref):
    npart=self.particles.size
    if npart == self.particles.size:
      names=list(particle_t.names)
      names.remove('s')
      general=0
      partn=1
      fmts="%-12s: %-14s %-14s %-14s %-14s"
      lgd=('Variable','Reference','Value','Difference','Relative Diff')
      lgds=True
      fmt=fmts.replace('-14s','14.6e')
      for pval,pref in zip(self.particles.flatten(),ref.particles.flatten()):
          pdiff=0
          for nn in names:
              val=pval[nn]
              ref=pref[nn]
              diff=ref-val
              if abs(diff)>0:
                  if abs(ref)>0:
                      rdiff=diff/ref
                  else:
                      rdiff=diff
                  if lgds:
                      print(fmts%lgd); lgds=False
                  print(fmt%(nn,ref,val,diff,rdiff))
                  pdiff+=rdiff**2
          if pdiff>0:
              pl='Part %d/%d'%(partn,npart)
              print("%-12s:  global diff  %14.6e"%(pl,np.sqrt(pdiff)))
              general+=pdiff
          partn+=1
      return general==0
    else:
      raise ValueError("Shape ref not compatible")



