import sys; sys.path.append('../')

import sixtracklib
import numpy as np

import scipy.optimize as so

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e6
lag_deg = 120
h_RF = 35000


import tfsdata

twdict = tfsdata.open('twiss.out')

machine = sixtracklib.CBlock()

p0c_eV = 450e9
gamma0 = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
beta0 = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)

length = twdict['param']['length'] 
f_RF = h_RF*c_light*beta0/(length)

# I want to start in a place with 0 dispersion
start_at = 'AT_IP5'
i_start = np.where(twdict['name']==start_at)[0][0]

indices = range(i_start, len(twdict['name']))+range(0, i_start)

for i_ele in indices:

	name = twdict['name'][i_ele]

	if twdict['keyword'][i_ele]=='MULTIPOLE':
		if twdict['k0l'][i_ele] != 0:
			machine.add_Multipole(name=name, knl=[twdict['k0l'][i_ele]], hxl=twdict['k0l'][i_ele], length=1e20)
		else:
			#print name
			machine.add_Multipole(name=name, knl=[0.,twdict['k1l'][i_ele],twdict['k2l'][i_ele]])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	elif twdict['keyword'][i_ele]=='RFCAVITY':
		print('Found cavity: '+name)
		machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_deg)
	else:
		print('Skipped: %s'%name)

#Add dipole corrector to distort the orbit
machine.add_Multipole(name=name, knl=[100.e-6])
machine.add_Multipole(name=name, ksl=[30.e-6])


# Build 1-turn map function
def one_turn_map(coord_in, flag_ebe=False):

	# coord = np.array([x, px, y, py, sigma, delta])

	coord = coord_in
	
	npart = 1
	
	bunch=sixtracklib.CParticles(npart=npart, 
			p0c=p0c_eV,
			beta0 = beta0,
			gamma0 = gamma0)
	bunch.x+=coord[0]
	bunch.px+=coord[1]
	bunch.y+=coord[2]
	bunch.py+=coord[3]
	bunch.sigma+=coord[4]
	bunch.set_delta(coord[5])

	particles,ebe,tbt = machine.track_cl(bunch,nturns=1,elembyelem={True:True, False:None}[flag_ebe], turnbyturn=True)

	coord =  np.array([tbt.x[1][0], tbt.px[1][0], tbt.y[1][0], tbt.py[1][0], 
					tbt.sigma[1][0], tbt.delta[1][0]])

	if flag_ebe:
		return coord, ebe
	else:
		return coord

# Define function for optimization
tominimize = lambda coord: np.sum((one_turn_map(coord)-coord)**2)

# Find fixed point
res = so.minimize(tominimize, np.array([0.,0.,0.,0.,0.,0.]), tol=1e-20, method='Nelder-Mead')

temp, ebe_CO = one_turn_map(res.x, flag_ebe=True)


# Track for many turns
npart = 1
delta_track = 1e-4
x_track = 1e-3
y_track = 2e-3
n_turns = 1000

n_iter = 10

bunch=sixtracklib.CParticles(npart=npart, 
	p0c=p0c_eV,
	beta0 = beta0,
	gamma0 = gamma0)
bunch.x = np.array([x_track])
bunch.y = np.array([y_track])
bunch.set_delta(delta_track)


for i_iter in range(n_iter):
	print('Segment %d/%d'%(i_iter, n_iter))
	# print bunch.x

	particles,ebe,tbt=machine.track_cl(bunch,nturns=n_turns,elembyelem=True,turnbyturn=True)

	# print particles.x

	if i_iter == 0:
		ebe_x_mean = np.sum(ebe.x[:,:,0], axis=0)
		ebe_px_mean = np.sum(ebe.px[:,:,0], axis=0)
		ebe_y_mean = np.sum(ebe.y[:,:,0], axis=0)
		ebe_py_mean = np.sum(ebe.py[:,:,0], axis=0)
		ebe_sigma_mean = np.sum(ebe.sigma[:,:,0], axis=0)
		ebe_delta_mean = np.sum(ebe.delta[:,:,0], axis=0)		
	else:
		ebe_x_mean = np.sum(ebe.x[:,:,0], axis=0)+ebe_x_mean
		ebe_px_mean = np.sum(ebe.px[:,:,0], axis=0)+ebe_px_mean
		ebe_y_mean = np.sum(ebe.y[:,:,0], axis=0)+ebe_y_mean
		ebe_py_mean = np.sum(ebe.py[:,:,0], axis=0)+ebe_py_mean
		ebe_sigma_mean = np.sum(ebe.sigma[:,:,0], axis=0)+ebe_sigma_mean
		ebe_delta_mean = np.sum(ebe.delta[:,:,0], axis=0)+ebe_delta_mean

	# I need to instantiate a new bunch otherwise I get segfault (to be investigated...)
	bunch=sixtracklib.CParticles(npart=npart, 
		p0c=p0c_eV,
		beta0 = beta0,
		gamma0 = gamma0)
	bunch.x = particles.x
	bunch.y = particles.y
	bunch.sigma = particles.sigma
	bunch.px = particles.px
	bunch.py = particles.py
	bunch.set_delta(particles.delta)

ebe_x_mean = ebe_x_mean/float(n_iter*n_turns)
ebe_px_mean = ebe_px_mean/float(n_iter*n_turns)
ebe_y_mean = ebe_y_mean/float(n_iter*n_turns)
ebe_py_mean = ebe_py_mean/float(n_iter*n_turns)
ebe_sigma_mean = ebe_sigma_mean/float(n_iter*n_turns)
ebe_delta_mean = ebe_delta_mean/float(n_iter*n_turns)


# closed orbit from mean
found_mean = np.array([ ebe_x_mean[-1], ebe_px_mean[-1],
						ebe_y_mean[-1], ebe_py_mean[-1],
						ebe_sigma_mean[-1], ebe_delta_mean[-1]])


print('Found mean:', found_mean)
print('Res opt:', res.x)

print('Val at mean:', tominimize(found_mean))
print('Val at res opt:', tominimize(res.x))



import matplotlib.pyplot as pl
pl.close('all')
fig1 = pl.figure(1, figsize=(8*2,6))
fig1.set_facecolor('w')
spx = pl.subplot(2,2,1)
spy = pl.subplot(2,2,3, sharex=spx)
spfx = pl.subplot(2,2,2)
spfy = pl.subplot(2,2,4, sharex=spfx)

fig2 = pl.figure(2, figsize=(8*2,6))
fig2.set_facecolor('w')
sps = pl.subplot(2,2,1)
spd = pl.subplot(2,2,3, sharex=spx)
spphase = pl.subplot2grid(shape=(2,2), loc=(0,1), rowspan=2)




import numpy.fft as fft

for i_part in range(npart):

	spec_x = fft.fft(tbt.x[:,i_part])
	freq = fft.fftfreq(len(spec_x))
	spec_y = fft.fft(tbt.y[:,i_part])

	spx.plot(tbt.x[:,i_part])
	# spx.plot(coord_mat[:,0])
	spy.plot(tbt.y[:,i_part])

	spfx.plot(freq, np.abs(spec_x))
	spfy.plot(freq, np.abs(spec_y))

	sps.plot(tbt.sigma[:,i_part])
	spd.plot(tbt.delta[:,i_part])

	spphase.plot(tbt.sigma[:,i_part], tbt.delta[:,i_part])


spphase.set_xlim(-.6, .6)
sps.set_ylim(-.6, .6)

spfx.set_xlim(left=-.05)

pl.figure(3, figsize=(8*2, 6))
spebe1 = pl.subplot(3,2,1)
spebe2 = pl.subplot(3,2,3, sharex=spebe1, sharey=spebe1)
spebe3 = pl.subplot(3,2,5, sharex=spebe1)
spebe1.plot(np.squeeze(ebe_CO.x))
spebe2.plot(np.squeeze(ebe_CO.y))
spebe3.plot(np.squeeze(ebe_CO.sigma))
spebe1.plot(ebe_x_mean)
spebe2.plot(ebe_y_mean)
spebe3.plot(ebe_sigma_mean)

spebep1 = pl.subplot(3,2,2)
spebep2 = pl.subplot(3,2,4, sharex=spebe1, sharey=spebep1)
spebep3 = pl.subplot(3,2,6, sharex=spebep1)
spebep1.plot(np.squeeze(ebe_CO.px))
spebep2.plot(np.squeeze(ebe_CO.py))
spebep3.plot(np.squeeze(ebe_CO.delta))
spebep1.plot(ebe_px_mean)
spebep2.plot(ebe_py_mean)
spebep3.plot(ebe_delta_mean)


for sp in [spebe1, spebe2, spebe3, spebep1, spebep2, spebep3]:
	sp.ticklabel_format(style='sci', scilimits=(0,0),axis='y')


# x_vect = np.linspace(-3e-2, 3e-2, 110)
# pl.figure(1000)
# pl.plot(x_vect, map(lambda x: tominimize(np.array([x, px_mean, 0, 0., 0., 0.])), x_vect))
# pl.axvline(x = x_mean)

# px_vect = np.linspace(-3e-4, 3e-4, 100)
# pl.figure(2000)
# pl.plot(px_vect, map(lambda px: tominimize(np.array([x_mean, px, 0, 0., 0., 0.])), px_vect))
# pl.axvline(x = px_mean)

# mat = np.zeros((len(x_vect), len(px_vect)))

# for ix, x in enumerate(x_vect):
# 	print ix
# 	for ipx, px in enumerate(px_vect):
# 		mat[ix, ipx] = tominimize(np.array([x, px, 0, 0., 0., 0.]))

# pl.figure(100)
# pl.pcolormesh(x_vect, px_vect, np.log10(mat).T)

pl.show()