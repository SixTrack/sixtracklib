import pickle
import pysixtrack
import numpy as np

import example_helpers as hp
import footprint


epsn_x = 2.e-6
epsn_y = 2.e-6
r_max_sigma = 10.
N_r_footp = 10.
N_theta_footp = 10.

with open('line.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid), keepextra=True)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pysixtrack.Particles.from_dict(pickle.load(fid))

part = partCO.copy() # pysixtrack.Particles(**partCO)
part._m = pysixtrack.Particles()._m # to be sorted out later

'''
# get beta functions from tracking
n_turns_beta = 150

# Track a particle to get betas
part.x += 1e-5
part.y += 1e-5

x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(
    line, part=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=0.,
    Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=0.,
    Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns_beta, verbose=True)


beta_x, x_max, px_cut = hp.betafun_from_ellip(x_tbt, px_tbt)
beta_y, y_max, py_cut = hp.betafun_from_ellip(y_tbt, py_tbt)

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
spx = fig1.add_subplot(2, 1, 1)
spy = fig1.add_subplot(2, 1, 2, sharex=spx)

spx.plot(x_tbt)
spy.plot(y_tbt)

fig2 = plt.figure(2)
spex = fig2.add_subplot(2, 1, 1)
spey = fig2.add_subplot(2, 1, 2)

spex.plot(x_tbt, px_tbt, '.')
spey.plot(y_tbt, py_tbt, '.')

spex.plot(0, px_cut, 'xr')
spey.plot(0, py_cut, 'xr')

plt.show()

'''

# get beta functions from twiss table
with open('twiss_at_start.pkl', 'rb') as fid:
    twiss_at_start = pickle.load(fid)

beta_x = twiss_at_start['betx']
beta_y = twiss_at_start['bety']

sigmax = np.sqrt(beta_x * epsn_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * epsn_y / part.beta0 / part.gamma0)

xy_norm = footprint.initial_xy_polar(r_min=5e-2, r_max=r_max_sigma, r_N=N_r_footp + 1,
                                     theta_min=np.pi / 100, theta_max=np.pi / 2 - np.pi / 100,
                                     theta_N=N_theta_footp)

DpxDpy_wrt_CO = np.zeros_like(xy_norm)

for ii in range(xy_norm.shape[0]):
    for jj in range(xy_norm.shape[1]):

        DpxDpy_wrt_CO[ii, jj, 0] = xy_norm[ii, jj, 0] * np.sqrt(epsn_x / part.beta0 / part.gamma0 / beta_x)
        DpxDpy_wrt_CO[ii, jj, 1] = xy_norm[ii, jj, 1] * np.sqrt(epsn_y / part.beta0 / part.gamma0 / beta_y)

with open('DpxDpy_for_footprint.pkl', 'wb') as fid:
    pickle.dump({
                'DpxDpy_wrt_CO': DpxDpy_wrt_CO,
                'xy_norm': xy_norm,
                }, fid)

