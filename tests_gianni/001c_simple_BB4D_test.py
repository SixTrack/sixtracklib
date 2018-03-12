import sys; sys.path.append('../')

import sixtracklib
import numpy as np

from scipy.constants import e as qe
from scipy.constants import c as c_light


q_part = qe
N_part = 1e15
sigma_x = 1e-3
sigma_y = 2e-3
beta_s = 1.
min_sigma_diff = 1e-16



machine = sixtracklib.CBlock()


machine.add_BeamBeam4D(name='bb4d',
    q_part = q_part,
    N_part = N_part,
    sigma_x = sigma_x,
    sigma_y = sigma_y,
    beta_s = beta_s,
    min_sigma_diff = min_sigma_diff)



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
    
