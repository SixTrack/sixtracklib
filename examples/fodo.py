import sixtracklib

fodo=sixtracklib.CBlock()
fodo.add_Drift(length=1.5)
fodo.add_Multipole(knl=[0.0,0.001])
fodo.add_Drift(length=1.5)
fodo.add_Multipole(name='qd',knl=[0.0,-0.001])

#fodo.add(stl.Drift(length=1.5))
#assert fodo.elem[3].knl == fodo.ns.qd.knl

bunch=sixtracklib.CParticles(npart=4)
bunch.x[1]=0.3
bunch.y[2]=0.2
bunch.z[3]=0.1

assert bunch.x[0]==0.3

el,tt=fodo.cl_track(bunch,nturns=1)
