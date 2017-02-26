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

block.track(cbeam,nturn=10000,turnbyturn=True)
bb=block.turnbyturn

#assert bnew[:10].compare(bref[:10],include=['s'])==0







