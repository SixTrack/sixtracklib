import sys
sys.path.append('../../../NAFFlib/')

import pickle
import pysixtrack
import numpy as np
import NAFFlib
import example_helpers as hp
import footprint
import matplotlib.pyplot as plt

track_with = 'PySixtrack'
#track_with = 'Sixtracklib'
device = '0.0'
device = None

n_turns = 100

with open('line.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid), keepextra=True)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pysixtrack.Particles.from_dict(pickle.load(fid))

with open('DpxDpy_for_footprint.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']

part = partCO.copy() # pysixtrack.Particles(**partCO)
part._m = pysixtrack.Particles()._m # to be sorted out later
part.sigma += 0.05

if track_with == 'PySixtrack':

    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(
        line, part=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0, Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns, verbose=True)

    info = track_with

elif track_with == 'Sixtracklib':
    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_sixtracklib(
        line=line, partCO=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')


# remove dispersive closed orbit from transverse coordinates
# Dx = np.mean(x_tbt*delta_tbt,axis=0)/np.mean(delta_tbt*delta_tbt,axis=0)
# Dpx = np.mean(px_tbt*delta_tbt,axis=0)/np.mean(delta_tbt*delta_tbt,axis=0)
# x_tbt -= Dx*delta_tbt
# px_tbt -= Dpx*delta_tbt
# Dy = np.mean(y_tbt*delta_tbt,axis=0)/np.mean(delta_tbt*delta_tbt,axis=0)
# Dpy = np.mean(py_tbt*delta_tbt,axis=0)/np.mean(delta_tbt*delta_tbt,axis=0)
# y_tbt -= Dy*delta_tbt
# py_tbt -= Dpy*delta_tbt
Dx = np.mean(x_tbt*delta_tbt)/np.mean(delta_tbt*delta_tbt)
Dpx = np.mean(px_tbt*delta_tbt)/np.mean(delta_tbt*delta_tbt)
x_tbt -= Dx*delta_tbt
px_tbt -= Dpx*delta_tbt
# Dy = np.mean(y_tbt*delta_tbt)/np.mean(delta_tbt*delta_tbt)
# Dpy = np.mean(py_tbt*delta_tbt)/np.mean(delta_tbt*delta_tbt)
# y_tbt -= Dy*delta_tbt
# py_tbt -= Dpy*delta_tbt


n_part = x_tbt.shape[1]
Qx = np.zeros(n_part)
Qy = np.zeros(n_part)

for i_part in range(n_part):
    Qx[i_part] = NAFFlib.get_tune(x_tbt[:, i_part])
    Qy[i_part] = NAFFlib.get_tune(y_tbt[:, i_part])

Qxy_fp = np.zeros_like(xy_norm)

Qxy_fp[:, :, 0] = np.reshape(Qx, Qxy_fp[:, :, 0].shape)
Qxy_fp[:, :, 1] = np.reshape(Qy, Qxy_fp[:, :, 1].shape)

plt.close('all')

fig3 = plt.figure(3, figsize=(5,5))
axcoord = fig3.add_subplot(1, 1, 1)
footprint.draw_footprint(xy_norm, axis_object=axcoord, linewidth = 1)
axcoord.set_xlim(right=np.max(xy_norm[:, :, 0]))
axcoord.set_ylim(top=np.max(xy_norm[:, :, 1]))
axcoord.set_xlabel('px (sigma)')
axcoord.set_ylabel('py (sigma)')

fig4 = plt.figure(4, figsize=(5,5))
axFP = fig4.add_subplot(1, 1, 1)
footprint.draw_footprint(Qxy_fp, axis_object=axFP, linewidth = 1)
# fig4.suptitle(info)
axFP.set_xlabel('Qx')
axFP.set_ylabel('Qy')
axFP.set_xlim(0,0.25)
axFP.set_ylim(0,0.25)
plt.show()

