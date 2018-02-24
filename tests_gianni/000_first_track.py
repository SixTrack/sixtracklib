import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e6
lag_RF = 0.
h_RF = 1


import tfsdata

twdict = tfsdata.open('twiss.out')

machine = sixtracklib.CBlock()

length = twdict['param']['length'] 
f_RF = h_RF*c_light/(length)

machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_RF)

for i_ele, name in enumerate(twdict['name']):
	if twdict['keyword'][i_ele]=='MULTIPOLE':
		if twdict['k0l'][i_ele] != 0:
			pass
			#machine.add_Multipole(name=name, knl=[twdict['k0l'][i_ele]], hxl=twdict['k0l'][i_ele], length=1e10)
		else:
			machine.add_Multipole(name=name, knl=[0.,twdict['k1l'][i_ele]])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	else:
		print('Skipped: %s'%name)




p0c_eV = 6500e9
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

delta = 1e-6
rpp = 1./(delta+1)
pc_eV = p0c_eV/rpp
gamma = np.sqrt(1. + (pc_eV/pmass_eV)**2)
beta = np.sqrt(1.-1./gamma**2)
rvv=beta/beta0
psigma = pmass_eV*(gamma-gamma0)/(beta0*p0c_eV)

bunch=sixtracklib.CParticles(npart=1, 
		p0c=p0c_eV,
		beta0 = beta0,
		gamma0 = gamma0,
		delta = delta,
		rvv = rvv,
		rpp = rpp,
		psigma = psigma)
bunch.x[0]=0.0
bunch.y[0]=0.0

particles,ebe,tbt=machine.track_cl(bunch,nturns=1024*10,elembyelem=None,turnbyturn=True)

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

pl.figure(3)
axl1 = pl.subplot(2,1,1)
pl.plot(tbt.delta[:,0])

axl2 = pl.subplot(2,1,2)
pl.plot(tbt.sigma[:,0])

pl.show()