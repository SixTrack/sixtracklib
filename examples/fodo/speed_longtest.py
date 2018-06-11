#!/usr/bin/env python
import os

import numpy as np

from pyoptics import madlang, MadBeam
import sixtracklib

def newer(f1,f2):
    if os.path.exists(f2):
        return os.path.getmtime(f1)>os.path.getmtime(f2)
    else:
        return True

mad=madlang.open('bench.madx')
line,rest=mad.bench.expand_struct()
if newer('bench.madx','track.obs0001.p0001'):
  os.system('madx bench.madx')

madbeam=MadBeam.open('track.obs0001.p0001','twiss_bench.tfs')

block=sixtracklib.cBlock.from_line(line)
bref=sixtracklib.cBeam.from_full_beam(madbeam.get_full_beam())
cbeam=bref.copy()[[0]]

import time
import sys
import pyopencl

def mkbench(npart,nturn):
  nturn=int(nturn)
  block=sixtracklib.cBlock.from_line(line)
  cbeam=bref.copy().reshape(-1)[:npart]
  st=time.time()
  block.track_cl(cbeam,nturn=nturn,turnbyturn=True)
  st=time.time()-st
  perfgpu=st/npart/nturn*1e9
  print("GPU part %4d, turn %4d: %10.3f nsec/part*turn"%(npart,nturn,perfgpu))

  block=sixtracklib.cBlock.from_line(line)
  npart2=npart/100
  cbeam=bref.copy().reshape(-1)[:npart2]
  st=time.time()
  block.track(cbeam,nturn=nturn,turnbyturn=True)
  st=time.time()-st
  perfcpu=st/npart2/nturn*1e9
  print("CPU part %4d, turn %4d: %10.3f nsec/part*turn"%(npart2,nturn,perfcpu))

  print("GPU/CPU : %g"%(perfcpu/perfgpu))
  return st,npart,nturn,perfgpu,perfcpu

out=open(time.strftime("bench_%Y%M%dT%H%m%S.txt"),'w')
out.write("#%s"%pyopencl.get_platforms()[0].get_devices()[0])
for npart in [100,1000,2000,5000,10000,20000]:
    for nturn in [1e5,2e5,10e5]:
        st,npart,nturn,perfgpu,perfcpu=mkbench(npart,nturn)
        fmt="%5d %5d %10.3f %10.3f %10.3f\n"
        out.write(fmt%(npart,nturn,perfgpu,perfcpu,perfcpu/perfgpu))
