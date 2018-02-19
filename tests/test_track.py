import numpy as np

import sixtracklib

def test_track():
  fodo=sixtracklib.CBlock()
  fodo.add_Drift(length=1.5)
  fodo.add_Multipole(knl=[0.0,0.001])
  fodo.add_Drift(length=1.3)
  fodo.add_Multipole(name='qd',knl=[0.0,-0.001])

  bunch=sixtracklib.CParticles(npart=4)
  bunch.x[1]=0.3
  bunch.y[2]=0.2
  bunch.sigma[3]=0.1

  particles,ebe,tbt=fodo.track_cl(bunch,1,True,True)
  return particles,ebe,tbt


if __name__=='__main__':
    particles,ebe,tbt=test_track()


