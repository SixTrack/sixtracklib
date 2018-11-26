import pickle
import pysixtrack
import numpy as np
import helpers as hp
import matplotlib.pyplot as plt

n_turns = 100

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

Dx_m = 1e-4
Dpx_rad = 1e-6
Dy_m = 2e-4
Dpy_rad = 3e-6
Dsigma_m = 2e-3
Ddelta = 0.

part = pysixtrack.Particles(**partCO)


print('Tracking PyST')
x_tbt_pyST, px_tbt_pyST, y_tbt_pyST, py_tbt_pyST, sigma_tbt_pyST, delta_tbt_pyST = hp.track_particle_pysixtrack(
    line, part=part, Dx_wrt_CO_m=Dx_m, Dpx_wrt_CO_rad=Dpx_rad,
    Dy_wrt_CO_m=Dy_m, Dpy_wrt_CO_rad=Dpy_rad,
    Dsigma_wrt_CO_m=Dsigma_m, Ddelta_wrt_CO=Ddelta, n_turns=n_turns, verbose=True)
print('Tracking ST')
x_tbt_ST, px_tbt_ST, y_tbt_ST, py_tbt_ST, sigma_tbt_ST, delta_tbt_ST = hp.track_particle_sixtrack(
    partCO=partCO, Dx_wrt_CO_m=np.array([Dx_m, Dx_m]), Dpx_wrt_CO_rad=Dpx_rad,
    Dy_wrt_CO_m=Dy_m, Dpy_wrt_CO_rad=Dpy_rad,
    Dsigma_wrt_CO_m=Dsigma_m, Ddelta_wrt_CO=Ddelta, n_turns=n_turns)
print('Tracking STlib')
x_tbt_STl, px_tbt_STl, y_tbt_STl, py_tbt_STl, sigma_tbt_STl, delta_tbt_STl = hp.track_particle_sixtracklib(
    line, partCO=partCO, Dx_wrt_CO_m=Dx_m, Dpx_wrt_CO_rad=Dpx_rad,
    Dy_wrt_CO_m=Dy_m, Dpy_wrt_CO_rad=Dpy_rad,
    Dsigma_wrt_CO_m=Dsigma_m, Ddelta_wrt_CO=Ddelta, n_turns=n_turns)

plt.close('all')
fig1 = plt.figure(1, figsize=(8 * 1.5, 6 * 1.2))
axx = fig1.add_subplot(3, 2, 1)
axx.plot(x_tbt_pyST[:, 0], label='PyST')
axx.plot(x_tbt_ST[:, 0], label='ST')
axx.plot(np.arange(len(x_tbt_STl[:, 0]))+1, x_tbt_STl[:, 0], label='STl')

axy = fig1.add_subplot(3, 2, 2, sharex=axx)
axy.plot(y_tbt_pyST)
axy.plot(y_tbt_ST[:, 0])
axy.plot(np.arange(len(x_tbt_STl[:, 0]))+1, y_tbt_STl[:, 0])

axpx = fig1.add_subplot(3, 2, 3, sharex=axx)
axpx.plot(px_tbt_pyST)
axpx.plot(px_tbt_ST[:, 0])
axpx.plot(np.arange(len(x_tbt_STl[:, 0]))+1, px_tbt_STl[:, 0])

axpy = fig1.add_subplot(3, 2, 4, sharex=axx)
axpy.plot(py_tbt_pyST)
axpy.plot(py_tbt_ST[:, 0])
axpy.plot(np.arange(len(x_tbt_STl[:, 0]))+1, py_tbt_STl[:, 0])

axsigma = fig1.add_subplot(3, 2, 5, sharex=axx)
axsigma.plot(sigma_tbt_pyST)
axsigma.plot(sigma_tbt_ST[:, 0])
axsigma.plot(np.arange(len(x_tbt_STl[:, 0]))+1, sigma_tbt_STl[:, 0])

axdelta = fig1.add_subplot(3, 2, 6, sharex=axx)
axdelta.plot(delta_tbt_pyST)
axdelta.plot(delta_tbt_ST[:, 0])
axdelta.plot(np.arange(len(x_tbt_STl[:, 0]))+1, delta_tbt_STl[:, 0])

plt.show()
