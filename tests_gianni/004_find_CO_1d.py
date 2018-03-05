import sys; sys.path.append('../')

import sixtracklib
import numpy as np

import scipy.optimize as so

from scipy.constants import c as c_light
pmass_eV = 938.272046e6

V_RF = 10e6
lag_RF = np.pi
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
			machine.add_Multipole(name=name, knl=[0.,twdict['k1l'][i_ele],twdict['k2l'][i_ele]/2.])
	elif twdict['keyword'][i_ele]=='DRIFT':
		machine.add_Drift(name=name, length=twdict['l'][i_ele])
	elif twdict['keyword'][i_ele]=='RFCAVITY':
		print('Found cavity: '+name)
		machine.add_Cavity(voltage=V_RF,frequency=f_RF,lag=lag_RF)
	else:
		print('Skipped: %s'%name)

#Add dipole corrector to distort the orbit
machine.add_Multipole(name=name, knl=[100.e-6])
#machine.add_Multipole(name=name, ksl=[30.e-6])


def one_turn_map(coord_in):

	# coord = np.array([x, px, y, py, sigma, delta])

	coord = coord_in

	npart = 1

	delta = np.array([0.])
	rpp = 1./(delta+1)
	pc_eV = p0c_eV/rpp
	gamma = np.sqrt(1. + (pc_eV/pmass_eV)**2)
	beta = np.sqrt(1.-1./gamma**2)
	rvv=beta/beta0
	psigma = pmass_eV*(gamma-gamma0)/(beta0*p0c_eV)

	bunch=sixtracklib.CParticles(npart=npart, 
			p0c=p0c_eV,
			beta0 = beta0,
			gamma0 = gamma0,
			delta = delta,
			rvv = rvv,
			rpp = rpp,
			psigma = psigma)
	bunch.x+=coord[0]
	bunch.px+=coord[1]
	# bunch.y+=coord[2]
	# bunch.py+=coord[3]
	# bunch.sigma+=coord[4]

	particles,ebe,tbt = machine.track_cl(bunch,nturns=1,elembyelem=None,turnbyturn=True)

	coord =  np.array([tbt.x[1][0], tbt.px[1][0]])

	return coord


# fxdpt = so.fixed_point(one_turn_map, np.array([0.,0.,0.,0.,0.,0.]))

tominimize = lambda c: (np.sum((one_turn_map(c)-c)**2))*1e1


print 'Start optimization'
res = so.minimize(tominimize, np.array([0.,0.]), tol=1e-20, method='Nelder-Mead')


npart = 1

delta = np.array([0e-4])
rpp = 1./(delta+1)
pc_eV = p0c_eV/rpp
gamma = np.sqrt(1. + (pc_eV/pmass_eV)**2)
beta = np.sqrt(1.-1./gamma**2)
rvv=beta/beta0
psigma = pmass_eV*(gamma-gamma0)/(beta0*p0c_eV)

bunch=sixtracklib.CParticles(npart=npart, 
		p0c=p0c_eV,
		beta0 = beta0,
		gamma0 = gamma0,
		delta = delta,
		rvv = rvv,
		rpp = rpp,
		psigma = psigma)
bunch.x+=0.00
bunch.y+=0.00

n_turns = 2048

particles,ebe,tbt=machine.track_cl(bunch,nturns=n_turns,elembyelem=True,turnbyturn=True)

# coord = np.array(6*[0.])
# coord_list = []
# for i_turn in xrange(n_turns):
# 	coord_list.append(coord)
# 	coord = one_turn_map(coord)

# coord_mat = np.array(coord_list)
# x_mean = np.mean(coord_mat[:,0])
# px_mean = np.mean(coord_mat[:,1])
# y_mean = np.mean(coord_mat[:,2])
# py_mean = np.mean(coord_mat[:,3])

x_mean = np.mean(tbt.x)
px_mean = np.mean(tbt.px)
y_mean = np.mean(tbt.y)
py_mean = np.mean(tbt.py)
sigma_mean = np.mean(tbt.sigma)
delta_mean = np.mean(tbt.delta)

# res2 = so.minimize(tominimize, np.array([x_mean,px_mean,y_mean, py_mean,sigma_mean,delta_mean]), tol=1e-20)
# res3 = so.minimize(tominimize, res.x, tol=1e-20)

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

pl.figure(3)
spebe1 = pl.subplot(3,1,1)
spebe2 = pl.subplot(3,1,2, sharex=spebe1, sharey=spebe1)
spebe3 = pl.subplot(3,1,3, sharex=spebe1)
spebe1.plot(np.mean(ebe.x[:,:,0], axis=0))
spebe2.plot(np.mean(ebe.y[:,:,0], axis=0))
spebe3.plot(np.mean(ebe.sigma[:,:,0], axis=0))

pl.figure(4)
spebep1 = pl.subplot(3,1,1)
spebep2 = pl.subplot(3,1,2, sharex=spebe1, sharey=spebe1)
spebep3 = pl.subplot(3,1,3, sharex=spebe1)
spebep1.plot(np.mean(ebe.px[:,:,0], axis=0))
spebep2.plot(np.mean(ebe.py[:,:,0], axis=0))
spebep3.plot(np.mean(ebe.delta[:,:,0], axis=0))

x_vect = np.linspace(-3e-2, 3e-2, 110)
pl.figure(1000)
pl.plot(x_vect, map(lambda x: tominimize(np.array([x, px_mean])), x_vect))
pl.axvline(x = x_mean)

px_vect = np.linspace(-3e-4, 3e-4, 100)
pl.figure(2000)
pl.plot(px_vect, map(lambda px: tominimize(np.array([x_mean, px])), px_vect))
pl.axvline(x = px_mean)

mat = np.zeros((len(x_vect), len(px_vect)))

for ix, x in enumerate(x_vect):
	print ix
	for ipx, px in enumerate(px_vect):
		mat[ix, ipx] = tominimize(np.array([x, px]))

pl.figure(100)
pl.pcolormesh(x_vect, px_vect, np.log10(mat).T)
pl.plot(x_mean, px_mean, 'or')
pl.plot(res.x[0], res.x[1], 'ob')


pl.show()