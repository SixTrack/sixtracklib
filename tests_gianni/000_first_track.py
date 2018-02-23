import sys; sys.path.append('../')
sys.path.append('../../pyoptics/')

import sixtracklib
import numpy as np

import tfsdata

twdict = tfsdata.open('twiss.out')

machine = sixtracklib.CBlock()

for i_ele, name in enumerate(twdict['name']):
	if twdict['keyword'][i_ele]=='MULTIPOLE':
		machine.add_Multipole(name=name, knl=[0.0,twdict['k1l'][i_ele]])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	else:
		print('Skipped: %s'%name)

bunch=sixtracklib.CParticles(npart=2)
bunch.x[0]=0.3
bunch.y[0]=0.2

particles,ebe,tbt=machine.track_cl(bunch,nturns=1024,elembyelem=None,turnbyturn=True)

import numpy.fft as fft
spec_x = fft.fft(tbt.x[:,0])
freq = fft.fftfreq(len(spec_x))
spec_y = fft.fft(tbt.y[:,0])


import matplotlib.pyplot as pl
pl.close('all')
pl.figure(1)
ax1 = pl.subplot(2,1,1)
pl.plot(tbt.x[:,0])
ax2 = pl.subplot(2,1,2, sharex=ax1)
pl.plot(tbt.y[:,0])

pl.figure(2)
axf1 = pl.subplot(2,1,1)
pl.plot(freq, np.abs(spec_x))
axf2 = pl.subplot(2,1,2, sharex=axf1)
pl.plot(freq, np.abs(spec_y))

pl.show()