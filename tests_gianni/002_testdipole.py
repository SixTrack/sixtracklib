import sys; sys.path.append('../')

import sixtracklib

import numpy as np

machine = sixtracklib.CBlock()
machine.add_Multipole(name='test_dip', knl=[1e-3], length=1., hxl=1e-3)


#from scipy.constants import c as c_light
pmass_eV = 938.272046e6
p0c_eV = 6500e9
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

delta = 0.#1e-3
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
bunch.x[0]=0.3
bunch.y[0]=0.2

particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=None,turnbyturn=True)