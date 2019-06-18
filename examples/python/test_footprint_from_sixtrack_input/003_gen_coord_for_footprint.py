import pickle
import pysixtrack
import numpy as np

import helpers as hp
import footprint


epsn_x = 3.5e-6
epsn_y = 3.5e-6
r_max_sigma = 6.
N_r_footp = 10.
N_theta_footp = 10.
n_turns_beta = 150


with open('line.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

part = pysixtrack.Particles(**partCO)

# Track a particle to get betas
part.x += 1e-5
part.y += 1e-5

x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(
    line, part=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=0.,
    Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=0.,
    Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns_beta, verbose=True)


beta_x, x_max, px_cut = hp.betafun_from_ellip(x_tbt, px_tbt)
beta_y, y_max, py_cut = hp.betafun_from_ellip(y_tbt, py_tbt)

sigmax = np.sqrt(beta_x * epsn_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * epsn_y / part.beta0 / part.gamma0)

xy_norm = footprint.initial_xy_polar(r_min=1e-2, r_max=r_max_sigma, r_N=N_r_footp + 1,
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
