import sys; sys.path.append('../')
sys.path.append('../../pyoptics/')


import sixtracklib

import metaclass as mtc

twob = mtc.twiss('twiss.out')

machine = sixtracklib.CBlock()

for i_ele, name in enumerate(twob.NAME):
	if twob.KEYWORD[i_ele]=='MULTIPOLE':
		machine.add_Multipole(name=name, knl=[0.0,twob.K1L[i_ele]])
	elif twob.KEYWORD[i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twob.L[i_ele])
	else:
		print('Skipped: %s'%name)

bunch=sixtracklib.CParticles(npart=2)
bunch.x[0]=0.3
bunch.y[1]=0.2

particles,ebe,tbt=machine.track_cl(bunch,nturns=1024,elembyelem=True,turnbyturn=True)




