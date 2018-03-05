import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e6
lag_deg = 180
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
			machine.add_Multipole(name=name, knl=[0.,twdict['k1l'][i_ele],twdict['k2l'][i_ele]])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	elif twdict['keyword'][i_ele]=='RFCAVITY':
		print('Found cavity: '+name)
		machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_deg)
	else:
		print('Skipped: %s'%name)




npart = 10

delta = np.linspace(0, 8e-4, npart)
rpp = 1./(delta+1)
pc_eV = p0c_eV/rpp
gamma = np.sqrt(1. + (pc_eV/pmass_eV)**2)
beta = np.sqrt(1.-1./gamma**2)
rvv=beta/beta0
psigma = pmass_eV*(gamma-gamma0)/(beta0*p0c_eV)

bunch=sixtracklib.CParticles(npart=npart, 
		p0c=p0c_eV,
		beta0 = beta0,
		gamma0 = gamma0,
		delta = delta,
		rvv = rvv,
		rpp = rpp,
		psigma = psigma)
bunch.x+=0.001
bunch.y+=0.002

particles,ebe,tbt=machine.track_cl(bunch,nturns=512,elembyelem=None,turnbyturn=True)


import matplotlib.pyplot as pl
pl.close('all')
fig1 = pl.figure(1, figsize=(8*2,6))
fig1.set_facecolor('w')
spx = pl.subplot(2,2,1)
spy = pl.subplot(2,2,3, sharex=spx)
spfx = pl.subplot(2,2,2)
spfy = pl.subplot(2,2,4, sharex=spfx)

fig2 = pl.figure(2, figsize=(8*2,6))
fig2.set_facecolor('w')
sps = pl.subplot(2,2,1)
spd = pl.subplot(2,2,3, sharex=spx)
spphase = pl.subplot2grid(shape=(2,2), loc=(0,1), rowspan=2)


import numpy.fft as fft

for i_part in range(npart):

	spec_x = fft.fft(tbt.x[:,i_part])
	freq = fft.fftfreq(len(spec_x))
	spec_y = fft.fft(tbt.y[:,i_part])

	spx.plot(tbt.x[:,i_part])
	spy.plot(tbt.y[:,i_part])

	spfx.plot(freq, np.abs(spec_x))
	spfy.plot(freq, np.abs(spec_y))

	sps.plot(tbt.sigma[:,i_part])
	spd.plot(tbt.delta[:,i_part])

	spphase.plot(tbt.sigma[:,i_part], tbt.delta[:,i_part])


spphase.set_xlim(-.6, .6)
sps.set_ylim(-.6, .6)

spfx.set_xlim(left=-.05)

# pl.figure(1)
# ax1 = pl.subplot(2,1,1)
# 
# ax2 = pl.subplot(2,1,2, sharex=ax1)
# pl.plot(tbt.y[:,i_part])

# pl.figure(2)
# axf1 = pl.subplot(2,1,1)
# pl.plot(freq, np.abs(spec_x))
# axf2 = pl.subplot(2,1,2, sharex=axf1)
# pl.plot(freq, np.abs(spec_y))

# pl.figure(3)
# axl1 = pl.subplot(2,1,1)
# pl.plot(tbt.delta[:,i_part])

# axl2 = pl.subplot(2,1,2)
# pl.plot(tbt.sigma[:,i_part])

# pl.figure(4)
# axl1 = pl.subplot(2,1,1)
# pl.plot(tbt.sigma[:,i_part], tbt.delta[:,i_part])


pl.show()