import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import e as qe
from scipy.constants import c as c_light

#crossing plane
alpha = 0.7
#crossing angle
phi = 0.8
#Intensity strong beam
N_part_tot = 1.1e15
#bunch length strong beam (assumed gaussian)
sigmaz = 0.075*100
# N slices
N_slices = 50
# Single particle properties
q_part = qe
# Minimum difference to fall on round
min_sigma_diff = 1e-16
threshold_singular = 1e-16

 
# strong beam shape at the IP (decoupled round beam)
(Sig_11_0, Sig_12_0, Sig_13_0, 
Sig_14_0, Sig_22_0, Sig_23_0, 
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
20e-06,  0.,  0.,
0., 0., 0.,
0., 20e-6, 0., 0.)



machine = sixtracklib.CBlock()


machine.add_BeamBeam6D(name='bb',
                q_part=q_part, 
                N_part_tot=N_part_tot, 
                sigmaz=sigmaz, 
                N_slices=N_slices, 
                min_sigma_diff=min_sigma_diff, 
                threshold_singular=threshold_singular,
                phi=phi, 
                alpha=alpha, 
                Sig_11_0=Sig_11_0,
                Sig_12_0=Sig_12_0, 
                Sig_13_0=Sig_13_0,
                Sig_14_0=Sig_14_0, 
                Sig_22_0=Sig_22_0, 
                Sig_23_0=Sig_23_0, 
                Sig_24_0=Sig_24_0, 
                Sig_33_0=Sig_33_0, 
                Sig_34_0=Sig_34_0, 
                Sig_44_0=Sig_44_0)

bb = machine.bb_data_list[0]

p0c_eV = 2e9
pmass_eV = 938.272046e6
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

bunch=sixtracklib.CParticles(npart=2, 
                        p0c=p0c_eV,
                        beta0 = beta0,
                        gamma0 = gamma0)


x = 1e-3
px = 50e-3
y = 2e-3
py = 27e-3
sigma = 3.
delta = 2e-4


bunch.x = bunch.x*0.+x
bunch.y = bunch.x*0.+y
bunch.px = bunch.x*0.+px
bunch.py = bunch.x*0.+py
bunch.sigma = bunch.x*0.+sigma
bunch.delta = bunch.x*0.+delta

particles,ebe,tbt=machine.track_cl(bunch,nturns=1,
                                  elembyelem=True,turnbyturn=True)

print('=======================')
print('particles.delta', particles.delta)
print('particles.psigma', particles.psigma)
print('particles.rpp', particles.rpp)
print('particles.rvv', particles.rvv)	

bunchtest=sixtracklib.CParticles(npart=2, 
                        p0c=p0c_eV,
                        beta0 = beta0,
                        gamma0 = gamma0)
bunchtest.set_delta(particles.delta) 

print('=======================')
print('bunchtest.delta', bunchtest.delta)
print('bunchtest.psigma', bunchtest.psigma)
print('bunchtest.rpp', bunchtest.rpp)
print('bunchtest.rvv', bunchtest.rvv)	
