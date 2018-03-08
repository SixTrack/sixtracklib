import sys; sys.path.append('../')

import sixtracklib
import numpy as np

import scipy.optimize as so

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e3
lag_deg = 120
h_RF = 35000


import tfsdata

twdict = tfsdata.open('twiss.out')

machine = sixtracklib.CBlock()

p0c_eV = 0.2e9
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

length = twdict['param']['length'] 
f_RF = h_RF*c_light*beta0/(length)

machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_deg)
	


# Track for many turns
npart = 1
delta_track = 1e-3
x_track = 1e-3
y_track = 2e-3
n_turns = 500
n_iter = 4

bunch=sixtracklib.CParticles(npart=npart, 
	p0c=p0c_eV,
	beta0 = beta0,
	gamma0 = gamma0)
bunch.x = np.array([x_track])
bunch.y = np.array([y_track])
bunch.set_delta(delta_track)

print('=======================')
print('bunch.delta', bunch.delta)
print('bunch.psigma', bunch.psigma)
print('bunch.sigma', bunch.sigma)
print('bunch.rpp', bunch.rpp)
print('bunch.rvv', bunch.rvv)

particles,ebe,tbt=machine.track_cl(bunch,nturns=n_turns,elembyelem=True,turnbyturn=True)
print('=======================')
print('particles.delta', particles.delta)
print('particles.psigma', particles.psigma)
print('particles.sigma', particles.sigma)
print('particles.rpp', particles.rpp)
print('particles.rvv', particles.rvv)	

bunch2=sixtracklib.CParticles(npart=npart, 
	p0c=p0c_eV,
	beta0 = beta0,
	gamma0 = gamma0)
bunch2.x = np.array([x_track])
bunch2.y = np.array([y_track])
bunch2.set_delta(particles.delta)

print('=======================')
print('bunch2.delta', bunch2.delta)
print('bunch2.psigma', bunch2.psigma)
print('bunch2.sigma', bunch2.sigma)
print('bunch2.rpp', bunch2.rpp)
print('bunch2.rvv', bunch2.rvv)	

