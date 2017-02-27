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
cbeam=bref.copy()[0]

block.track(cbeam,nturn=1,elembyelem=True)

bnew=block.elembyelem[:,0][iconv]

for pp in range(len(bnew.x)):
    res=bnew[pp].compare(bref[pp],include=['s'],verbose=False)
    if res>1e-6:
    #if bnew[pp].psigma!=bnew[pp].psigma:
      print pp,names2[pp]
      bnew[pp].compare(bref[pp],include=['s'])
      break

import time
nturn=20;npart=5000
cbeam=bref.copy().reshape(-1)[:npart]
st=time.time()
block.track_cl(cbeam,nturn=nturn,turnbyturn=True)
st=time.time()-st
perfgpu=st/npart/nturn*1e3
print("GPU  part %6d %6d: %g msec/part*turn"%(npart,nturn,perfgpu))

cbeam=bref.copy().reshape(-1)[:npart]
st=time.time()
block.track(cbeam,nturn=nturn,turnbyturn=True)
st=time.time()-st
perfcpu=st/npart/nturn*1e3
print("CPU  part %6d %6d: %g msec/part*turn"%(npart,nturn,perfcpu))

print("GPU/CPU : %g"%(perfcpu/perfgpu))



#bb=block.turnbyturn

#assert bnew[:10].compare(bref[:10],include=['s'])==0







