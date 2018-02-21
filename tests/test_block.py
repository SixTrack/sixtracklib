import numpy as np

import sixtracklib

def test_add_drift():
    blk=sixtracklib.CBlock()
    blk.add_Drift(length=3)
    blk.add_Drift(length=2.5)
    assert blk.elem_ids==[0,2]

def test_add_fodo():
    fodo=sixtracklib.CBlock()
    fodo.add_Drift(length=1.5)
    fodo.add_Multipole(knl=[0.0,0.001])
    fodo.add_Drift(length=1.3)
    fodo.add_Multipole(name='qd',knl=[0.0,-0.001])
    fodo.add_Cavity(voltage=3.,frequency=4.,lag=5)
    fodo.add_Align(dx=1.,dy=-2.,tilt=30)
    assert fodo.elem[0].length==1.5
    assert fodo.elem['qd'][0].bal[2]==-0.001

def test_particle():
    bunch=sixtracklib.CParticles(npart=4)
    bunch.x[1]=0.3
    bunch.y[2]=0.2
    bunch.sigma[3]=0.1
    assert len(bunch.partid)==4
    assert bunch.sigma[3]==0.1
