import numpy as np
import matplotlib.pyplot as plt

import sixtracklib


block=sixtracklib.cBlock(2)
kqf=0.8; kqd=-0.7;ld=1.5;
block.Drift(ld)
block.Multipole([0.,kqf],[0.,0.],0,0,0)
block.Drift(ld)
block.Multipole([0.,kqd],[0.,0.],0,0,0)
block.Block()


npart=6
beam=sixtracklib.cBeam(npart)
beam.particles['x'][:npart/2]=np.linspace(0,1,npart/2)
beam.particles['y'][npart/2:]=np.linspace(0,1,npart/2)

#plt.clf()
#plt.plot(beam.particles['x'],beam.particles['px'])
#plt.plot(beam.particles['y'],beam.particles['py'])
block.track(beam,elembyelem=True)
#plt.plot(beam.particles['x'],beam.particles['px'])
#plt.plot(beam.particles['y'],beam.particles['py'])





