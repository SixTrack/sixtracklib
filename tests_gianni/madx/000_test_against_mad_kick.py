import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p, c

intensity_pbun = 1e11
energy_GeV = 7000.
nemittx = 2.5e-6
nemitty = 1.e-6

gamma_tr = 55.
#ap  = 1./(gamma_tr**2)
ap = 0.

hRF = 35640
fRF = 400e6

tune_x = 0.
tune_y = 0.
beta_s = 0.40


include_beambeam = True

offsetx_s = 5e-5
offsety_s = 0.
nturns =1
#offsetx_s = 0.
#offsety_s = 0.

#compute sigmas
mp_GeV = m_p*c**2/qe/1e9
gamma0 = energy_GeV/mp_GeV
beta0=np.sqrt(1.-1./gamma0**2)
sigmax_s = np.sqrt(beta_s*nemittx/gamma0/beta0)
sigmay_s = np.sqrt(beta_s*nemitty/gamma0/beta0)

sigma_avg = 0.5*(sigmax_s + sigmay_s)

theta_obs = np.pi/10
#~ theta_obs = np.pi/2.
#~ theta_obs = np.pi/3.


# set initial conditions
n_points = 100
r0_particles = np.array([0.] + list(np.linspace(-15*sigma_avg, 15*sigma_avg, n_points)))
x0_particles = r0_particles*np.cos(theta_obs)
px0_particles = 0*x0_particles
y0_particles = r0_particles*np.sin(theta_obs)
py0_particles = 0*x0_particles


    

import sys, os
BIN = os.path.expanduser("../../")
sys.path.append(BIN)
import sixtracklib


beam=sixtracklib.CParticles(npart=len(x0_particles), 
                        p0c=mp_GeV*1e9*gamma0*beta0,
                        beta0 = beta0,
                        gamma0 = gamma0)

beam.x = x0_particles
beam.px = px0_particles
beam.y = y0_particles
beam.py = py0_particles
beam.set_delta(0.)


machine = sixtracklib.CBlock()
machine.add_LinearMap(qx=tune_x,qy=tune_y,
            betax=beta_s,betay=beta_s, 
            alfax=0.,alfay=0.,
            ap=ap,h=hRF, fRF=fRF)

machine.add_BeamBeam4D(name='bb4d',
    q_part = qe,
    N_part = intensity_pbun,
    sigma_x = sigmax_s,
    sigma_y = sigmay_s,
    beta_s = beta0,
    min_sigma_diff = 1e-10,
    Delta_x = offsetx_s,
    Delta_y = offsety_s )
    
particles,ebe,tbt=machine.track_cl(beam,nturns=nturns,elembyelem=True,turnbyturn=True)

#Remove Dipole kick
kick_x = beam.px[1:]-beam.px[0] #we are building the vectors so that the first particle is the ref particle
kick_y = beam.py[1:]-beam.py[0]


  
### Tracking MAD
import track_mad as tm
_, _, px_particles_mad, py_particles_mad = tm.track_mad_linmap_and_beambeam(intensity_pbun, energy_GeV, nemittx, nemitty, 
                    tune_x, tune_y, beta_s, include_beambeam, offsetx_s, offsety_s, sigmax_s, sigmay_s, 
                    x0_particles, px0_particles, y0_particles, py0_particles, nturns=nturns)
kick_x_mad = px_particles_mad[1:, nturns]
kick_y_mad = py_particles_mad[1:, nturns]
import pylab as pl
pl.close('all')
pl.figure(1)
ax1 = pl.subplot(2,1,1)
pl.plot(r0_particles[1:], kick_x_mad, 'b', label='mad')
pl.plot(r0_particles[1:], kick_x, '.r', label='sixtracklib')
pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
pl.legend(loc='best')

pl.grid('on')
pl.subplot(2,1,2, sharex = ax1)
pl.plot(r0_particles[1:], kick_y_mad, 'b')
pl.plot(r0_particles[1:], kick_y, '.r')
pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='x') 
pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y') 

pl.suptitle('theta = %.1f deg'%(theta_obs*90/(np.pi/2)))
pl.grid('on')
pl.show()

