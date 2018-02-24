import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e6
lag_RF = np.pi
h_RF = 35000


import tfsdata

twdict = tfsdata.open('twiss.out')

machine = sixtracklib.CBlock()

p0c_eV = 450e9
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

length = twdict['param']['length'] 
f_RF = h_RF*c_light*beta0/(length)

# I want to start in a place with 0 dispersion
start_at = 'AT_IP5'
i_start = np.where(twdict['name']==start_at)[0][0]

indices = range(i_start, len(twdict['name']))+range(0, i_start)

for i_ele in indices:

	name = twdict['name'][i_ele]

	if twdict['keyword'][i_ele]=='MULTIPOLE':
		if twdict['k0l'][i_ele] != 0:
			machine.add_Multipole(name=name, knl=[twdict['k0l'][i_ele]], hxl=twdict['k0l'][i_ele], length=1e20)
		else:
			#print name
			machine.add_Multipole(name=name, knl=[0.,twdict['k1l'][i_ele],twdict['k2l'][i_ele]/2.])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	elif twdict['keyword'][i_ele]=='RFCAVITY':
		print('Found cavity: '+name)
		machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_RF)
	else:
		print('Skipped: %s'%name)





delta = 3e-4
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
bunch.x[0]=0.001
bunch.y[0]=0.002

particles,ebe,tbt=machine.track_cl(bunch,nturns=512,elembyelem=None,turnbyturn=True)

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

pl.figure(4)
axl1 = pl.subplot(2,1,1)
pl.plot(tbt.sigma[:,0], tbt.delta[:,0])


pl.show()