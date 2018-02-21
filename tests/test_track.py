import numpy as np

import sixtracklib

def test_track():
  fodo=sixtracklib.CBlock()
  fodo.add_Drift(length=1.5)
  fodo.add_Multipole(knl=[0.0,0.001])
  fodo.add_Drift(length=1.3)
  fodo.add_Multipole(name='qd',knl=[0.0,-0.001])

  bunch=sixtracklib.CParticles(npart=2)
  bunch.x[0]=0.3
  bunch.y[1]=0.2

  particles,ebe,tbt=fodo.track_cl(bunch,nturns=1,
                                  elembyelem=True,turnbyturn=True)
  return fodo,particles,ebe,tbt


if __name__=='__main__':
    fodo,particles,ebe,tbt=test_track()


