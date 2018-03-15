import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p, c

intensity_pbun = 1e11
energy_GeV = 7000.
nemittx = 2.5e-6
nemitty = 1.e-6

gamma_tr = 55.
#### Switch off longitudinal motion in linear map
#ap = 1./gamma_tr**2
ap = 0.
hRF = 35640
fRF = 400e6

tune_x = 0.31
tune_y = 0.32
beta_s = 0.40

nturns  = 1024

include_beambeam = True
offsetx_s = 5.e-5
#offsetx_s = 0.0
offsety_s = 0.0


n_sigma_footprint = 7
n_theta_footprint = 7

i_sigma_vect = np.arange(0.01, n_sigma_footprint+0.1)
theta_vect = np.linspace(0+np.pi/900, n_theta_footprint-np.pi/900, n_theta_footprint)

# set initial conditions
x0_particles = []
px0_particles = []
y0_particles = []
py0_particles = []

mp_GeV = m_p*c**2/qe/1e9
gamma0 = energy_GeV/mp_GeV
beta0=np.sqrt(1.-1./gamma0**2)
#mp_GeV = m_p*c**2/qe/1e9
#gamma = energy_GeV/mp_GeV

sigmax_s = np.sqrt(beta_s*nemittx/gamma0/beta0)
sigmay_s = np.sqrt(beta_s*nemitty/gamma0/beta0)


x0_particles = []
px0_particles = []
y0_particles = []
py0_particles = []

for i_sigma in i_sigma_vect:
  for theta in theta_vect:
    x0_particles.append(i_sigma*sigmax_s*np.cos(theta))
    y0_particles.append(i_sigma*sigmay_s*np.sin(theta))
    
    px0_particles.append(0.)
    py0_particles.append(0.)  


import sys, os
BIN = os.path.expanduser("../../")
sys.path.append(BIN)
import sixtracklib

### Build beam
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


x_particles_stlb = tbt.x.T
y_particles_stlb = tbt.y.T
mean_x = x_particles_stlb.mean(axis=1, keepdims=True)
mean_y = y_particles_stlb.mean(axis=1, keepdims=True)
x_particles_stlb -= mean_x
y_particles_stlb -= mean_y

### Tracking MAD
import track_mad as tm
x_particles_mad, y_particles_mad, _, _ = tm.track_mad_linmap_and_beambeam(intensity_pbun, energy_GeV, nemittx, nemitty, 
                    tune_x, tune_y, beta_s, include_beambeam, offsetx_s, offsety_s, sigmax_s, sigmay_s, 
                    x0_particles, px0_particles, y0_particles, py0_particles, nturns)
                    
### Tune analysis
import harmonic_analysis as ha
def extract_tune(signals): 
    n_signals = signals.shape[0]
    tune_list = []
    for i_signal in xrange(n_signals):
      tune_list.append(ha.HarmonicAnalysis(signals[i_signal, :]).laskar_method(num_harmonics=1)[0][0])
    return np.array(tune_list)

tunes_x_stlb = extract_tune(x_particles_stlb)
tunes_y_stlb = extract_tune(y_particles_stlb)

tunes_x_mad = extract_tune(x_particles_mad)
tunes_y_mad = extract_tune(y_particles_mad)

diff_x = [abs(x-y) for x,y in zip(tunes_x_stlb,tunes_x_mad)]
diff_y = [abs(x-y) for x,y in zip(tunes_y_stlb,tunes_y_mad)]

import pylab as pl
pl.close('all')
pl.figure(2)
pl.semilogy(diff_x, c='k',label='x-plane')
pl.semilogy(diff_y, c='b', label = 'y-plane')
pl.xlabel('# particles', fontsize=14)
pl.ylabel(r'$ \rm log(|Q_{MADX}-Q_{sixtracklib}|)$', fontsize=14)
pl.title('With offset')
pl.legend(loc=1)
pl.grid()
pl.tight_layout()
pl.figure(1)
pl.plot(tunes_x_mad, tunes_y_mad, '.r', label='mad')
pl.plot(tunes_x_stlb, tunes_y_stlb, 'xb', label='sixtracklib')
pl.suptitle('sigmax = %.2e sigmay = %.2e'%(sigmax_s, sigmay_s))
#~ pl.plot([.3, .33], [.3, .33], 'k')
pl.axis('equal')
pl.grid('on')
pl.legend(loc='best')
pl.show()

