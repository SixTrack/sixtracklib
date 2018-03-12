import numpy as np
from scipy.constants import e as qe
from scipy.constants import m_p, c

intensity_pbun = 1e11
energy_GeV = 7000.
nemittx = 2.5e-6
nemitty = 1.e-6

tune_x = 0.31
tune_y = 0.32
beta_s = 0.40

nturns  = 1024

include_beambeam = True
offsetx_s = 0.
offsety_s = 0.


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
gamma = energy_GeV/mp_GeV

sigmax_s = np.sqrt(beta_s*nemittx/gamma)
sigmay_s = np.sqrt(beta_s*nemitty/gamma)


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
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)
import sixtracklib


### Build beam
beam=sixtracklib.cBeam(npart=len(x0_particles))
for ii in xrange(len(beam.particles)):
  beam.particles[ii]['partid'] = ii
  beam.particles[ii]['elemid'] = 0
  beam.particles[ii]['turn'] = 0
  beam.particles[ii]['state'] = 0
  beam.particles[ii]['s'] = 0
  beam.particles[ii]['x'] = x0_particles[ii]
  beam.particles[ii]['px'] = px0_particles[ii]
  beam.particles[ii]['y'] = y0_particles[ii]
  beam.particles[ii]['py'] = py0_particles[ii]
  beam.particles[ii]['sigma'] = 0.
  beam.particles[ii]['psigma'] = 0.
  beam.particles[ii]['chi'] = 1.
  beam.particles[ii]['delta'] = 0.
  beam.particles[ii]['rpp'] = 1.
  beam.particles[ii]['rvv'] = 1.
  beam.particles[ii]['beta'] = 1.
  beam.particles[ii]['gamma'] = gamma
  beam.particles[ii]['mass0'] = m_p*c**2/qe
  beam.particles[ii]['charge0'] = qe
  beam.particles[ii]['beta0'] = 1.
  beam.particles[ii]['gamma0'] = gamma
  beam.particles[ii]['p0c'] = energy_GeV*1e9
  
###  Build the ring
block=sixtracklib.cBlock(size=50)
#block.Multipole(bal=np.array([1.,2.,3.,4.,5.,6.]),l=0,hx=0,hy=0)
block.LinMap(alpha_x_s0=0., beta_x_s0=beta_s, D_x_s0=0., 
             alpha_x_s1=0., beta_x_s1=beta_s, D_x_s1=0.,
             alpha_y_s0=0., beta_y_s0=beta_s, D_y_s0=0.,
             alpha_y_s1=0., beta_y_s1=beta_s, D_y_s1=0.,
             dQ_x=tune_x, dQ_y=tune_y)
             
if include_beambeam:
  if np.abs((sigmax_s-sigmay_s)/((sigmax_s+sigmay_s)/2.))<1e-3:
    print "round beam"
    block.BB4D(N_s = intensity_pbun, beta_s = beta_s, q_s = qe, 
              transv_field_data = {'type':'gauss_round',  'sigma': (sigmax_s+sigmay_s)/2., 'Delta_x': offsetx_s, 'Delta_y': offsety_s})             
  else:
    print "elliptic beam"
    block.BB4D(N_s = intensity_pbun, beta_s = beta_s, q_s = qe, transv_field_data = {'type':'gauss_ellip',  'sigma_x': sigmax_s, 'sigma_y': sigmay_s, 'Delta_x': offsetx_s, 'Delta_y': offsety_s})             

block.Block() 


### Tracking stlb
track_fun =  block.track
# test OPENCL:
# track_fun =  block.track_cl


x_particles_stlb = []
y_particles_stlb = []
for i in xrange(nturns):
  x_particles_stlb.append(beam.particles['x'].copy())
  y_particles_stlb.append(beam.particles['y'].copy())
  if i%100 == 0:
    print 'turn, ',i
  track_fun(beam)
x_particles_stlb = np.array(x_particles_stlb).T
y_particles_stlb = np.array(y_particles_stlb).T
  
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

import pylab as pl
pl.close('all')
pl.figure(1)
pl.plot(tunes_x_mad, tunes_y_mad, '.r', label='mad')
pl.plot(tunes_x_stlb, tunes_y_stlb, 'xb', label='sixtracklib')
pl.suptitle('sigmax = %.2e sigmay = %.2e'%(sigmax_s, sigmay_s))
#~ pl.plot([.3, .33], [.3, .33], 'k')
pl.axis('equal')
pl.grid('on')
pl.legend(loc='best')
pl.show()
