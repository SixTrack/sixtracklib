import numpy as np
import sixtracklib


block=sixtracklib.cBlock(2)
block.Multipole([1.,3.,5.],[2.,4.,6.],0,0,0,)
block.Drift(56.)
block.Drift(5.)
block.Block()

def test_track():
  beam=sixtracklib.cBeam(50)
  block.track(beam)

  assert beam.particles[2]['s']  ==61.0
  assert beam.particles[2]['px'] ==-1.0
  assert beam.particles[2]['py'] ==2.0

def test_track_cl():
  if hasattr(block,'track_cl'):
    beam=sixtracklib.cBeam(50)
    block.track_cl(beam)

    assert beam.particles[2]['s'] ==61.0
    assert beam.particles[2]['px']==-1.0
    assert beam.particles[2]['py']==2.0


