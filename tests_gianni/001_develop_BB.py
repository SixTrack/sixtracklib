import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import e as qe

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


machine.add_BeamBeam(name='bb',
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

bunch=sixtracklib.CParticles(npart=2)
bunch.x[0]=0.3
bunch.y[0]=0.2

particles,ebe,tbt=machine.track_cl(bunch,nturns=1,
                                  elembyelem=True,turnbyturn=True)
