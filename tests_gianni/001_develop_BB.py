import sys; sys.path.append('../')

import sixtracklib

import numpy as np

machine = sixtracklib.CBlock()

data=np.array([1,2,3,4],dtype='float')
machine.add_BeamBeam(name='bb',datasize=len(data),data=data)

bunch=sixtracklib.CParticles(npart=2)
bunch.x[0]=0.3
bunch.y[0]=0.2

particles,ebe,tbt=machine.track_cl(bunch,nturns=1,
                                  elembyelem=True,turnbyturn=True)
