import numpy as np

import sys
sys.path.insert(0, '../../')
import sixtracklib

from scipy.constants import e as qelem


# Build the machine

block=sixtracklib.cBlock(size=50)
#block.Multipole(bal=np.array([1.,2.,3.,4.,5.,6.]),l=0,hx=0,hy=0)
block.LinMap(alpha_x_s0=0., beta_x_s0=100., D_x_s0=0., 
             alpha_x_s1=0., beta_x_s1=100., D_x_s1=0.,
             alpha_y_s0=0., beta_y_s0=200., D_y_s0=0.,
             alpha_y_s1=0., beta_y_s1=200., D_y_s1=0.,
             dQ_x=0.31, dQ_y=0.32)
             
            
sigma_bb = 1e-3
block.BB4D(N_s = 1e11*100, beta_s = 1.0, q_s = qelem, transv_field_data = {'type':'gauss_round',  'sigma': sigma_bb, 'Delta_x': 1.*sigma_bb, 'Delta_y': 0.})             
block.BB4D(N_s = 1e11*100, beta_s = 1.0, q_s = qelem, transv_field_data = {'type':'gauss_ellip',  'sigma_x': sigma_bb*(1+.0001), 'sigma_y': sigma_bb*(1-.0001), 'Delta_x': 1.*sigma_bb, 'Delta_y': 0.})             

block.Block()

# Setup simulation

# test CPU
#track_fun =  block.track

# test OPENCL:
track_fun =  block.track_cl

# test CUDA:
# to be finalized :-P

beam=sixtracklib.cBeam(npart=50)
for ii in xrange(len(beam.particles)):
  beam.particles[ii]['partid'] = ii
  beam.particles[ii]['elemid'] = 0
  beam.particles[ii]['turn'] = 0
  beam.particles[ii]['state'] = 0
  beam.particles[ii]['s'] = 0
  beam.particles[ii]['x'] = float(ii)/float(beam.npart)*1e-2
  beam.particles[ii]['px'] = 0.
  beam.particles[ii]['y'] = float(ii)/float(beam.npart)*1e-2
  beam.particles[ii]['py'] = 0.
  beam.particles[ii]['sigma'] = 0.
  beam.particles[ii]['psigma'] = 0.
  beam.particles[ii]['chi'] = 1.
  beam.particles[ii]['delta'] = 0.
  beam.particles[ii]['rpp'] = 1.
  beam.particles[ii]['rvv'] = 1.
  beam.particles[ii]['beta'] = 1.
  beam.particles[ii]['gamma'] = 7000.
  beam.particles[ii]['mass0'] = 938e6
  beam.particles[ii]['charge0'] = qelem
  beam.particles[ii]['beta0'] = 1.
  beam.particles[ii]['gamma0'] = 7000.
  beam.particles[ii]['p0c'] = 7e12


# track
N_turns = 10000

i_part = beam.npart-1

x, px = [], []
y, py = [], []
for i in range(N_turns):
  x.append(beam.particles[2]['x'])
  px.append(beam.particles[2]['px'])
  y.append(beam.particles[2]['y'])
  py.append(beam.particles[2]['py'])
  print i
  track_fun(beam)


# Some plots
import pylab as plt
plt.close('all')
plt.figure(1, figsize=(8*1.5, 6))
ax1 = plt.subplot(1,2,1)
ax1.plot(x, px, '.') 


ax2 = plt.subplot(1,2,2)
ax2.plot(y, py, '.') 

for ax in [ax1, ax2]:
    ax.ticklabel_format(style='sci', scilimits=(0,0),axis='x')  
    ax.ticklabel_format(style='sci', scilimits=(0,0),axis='y')  

plt.show()


#~ print(beam.particles[2]['s'])
#~ print(beam.particles[2]['px'])
#~ print(beam.particles[2]['py'])


#~ beam=sixtracklib.cBeam(npart=50)
#~ track_fun(beam)

#~ print(beam.particles[2]['s'])
#~ print(beam.particles[2]['px'])
#~ print(beam.particles[2]['py'])











