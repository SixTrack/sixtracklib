#!/usr/bin/env python
import numpy as np

import sixtracktools
import sixtracklib


six =sixtracktools.SixTrackInput('.')
line,rest,iconv=six.expand_struct()
names,types,args=zip(*line)
idx=dict( (nn,ii) for ii,nn in enumerate(six.struct) if not 'BLOC' in nn)
names2=np.array(names)[iconv]


sixtrackbeam=sixtracktools.SixDump3('dump3.dat')

block=sixtracklib.cBlock.from_line(line)
bref=sixtracklib.cBeam.from_full_beam(sixtrackbeam.get_full_beam())
bref=bref.reshape(-1,2)

import time
import sys
npart=int(sys.argv[1]);
nturn=int(sys.argv[2]);
cbeam=bref.copy().reshape(-1)[:npart]
st=time.time()
block.track_cl(cbeam,nturn=nturn,turnbyturn=True)
st=time.time()-st
perfgpu=st/npart/nturn*1e3
print("GPU part %4d, turn %4d: %10.3f msec/part*turn"%(npart,nturn,perfgpu))

cbeam=bref.copy().reshape(-1)[:npart]
st=time.time()
block.track(cbeam,nturn=nturn,turnbyturn=True)
st=time.time()-st
perfcpu=st/npart/nturn*1e3
print("CPU part %4d, turn %4d: %10.3f msec/part*turn"%(npart,nturn,perfcpu))

print("GPU/CPU : %g"%(perfcpu/perfgpu))









