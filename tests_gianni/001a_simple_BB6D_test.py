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

 
#~ # strong beam shape at the IP (decoupled round beam)
#~ (Sig_11_0, Sig_12_0, Sig_13_0, 
#~ Sig_14_0, Sig_22_0, Sig_23_0, 
#~ Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
#~ 20e-06,  0.,  0.,
#~ 0., 0., 0.,
#~ 0., 20e-6, 0., 0.)

# strong beam shape at the IP (coupled beam)
(Sig_11_0, Sig_12_0, Sig_13_0, 
Sig_14_0, Sig_22_0, Sig_23_0, 
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
  8.4282060230000004e-06,  1.8590458800000001e-07,  -3.5512334410000001e-06,
 -3.8254462239999997e-08, 4.101510281e-09, -7.5517657920000006e-08,
 -8.1134615060000002e-10, 1.031446898e-05, 1.177863077e-07, 1.3458251810000001e-09)

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

p0c_eV = 6.5e12
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

#For comparison
print('\n\n\n')
names_list = 'x px y py sigma delta'.split()
for name in names_list:
    print('D_'+name+ ' %.10e'%np.diff(getattr(tbt, name), axis=0)[0,0])
    
