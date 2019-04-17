import numpy as np
import matplotlib.pyplot as pl
from cpymad.madx import Madx
import pysixtracklib as pyst


# prepare madx
mad=Madx()
mad.options.echo=False
mad.call(file="SPS_Q20_thin.seq")
mad.use(sequence='sps')
twiss=mad.twiss()
q1mad=twiss.summary['q1']
q2mad=twiss.summary['q2']

# Build elements for SixTrackLib
elements=pyst.Elements.from_mad(mad.sequence.sps)
nturns=2**14
bpm=elements.BeamMonitor(num_stores=nturns)

# Track one turn
npart=10
particles=pyst.Particles.from_ref(npart,p0c=26e6)
particles.x += np.linspace(0,1e-6,npart)
job = pyst.TrackJob(elements, particles,until_turn_elem_by_elem=1)
job.track_elem_by_elem(1)
pl.plot(job.output.particles[0].x[1::10])

# Track many turns CPU
npart=10
particles=pyst.Particles.from_ref(npart,p0c=26e6)
particles.x += np.linspace(0,1e-6,npart)
job = pyst.TrackJob(elements, particles)
job.track(nturns)

# Find tunes
ff=np.linspace(0,0.5,nturns//2+1)
x=job.output.particles[0].x[1::npart]
xf=abs(np.fft.rfft(x))
plot(ff,xf)
q1st=ff[xf.argmax()]
print((q1mad-20)-q1st)

# Track many particles few turns GPU
npart=5000; nturns=256
bpm.num_stores=nturns
particles=pyst.Particles.from_ref(npart,p0c=26e6)
particles.x += np.linspace(0,1e-1,npart)
job = pyst.TrackJob(elements, particles,device="opencl:0.0")
job.track(nturns)
job.collect()

x=job.output.particles[0].x.copy()
x=x.reshape(nturns,npart)
pl.imshow(abs(np.fft.rfft(x,axis=0)),aspect='auto')

px=job.output.particles[0].px.copy()
px=px.reshape(nturns,npart)

for i in range(npart):
    plot(x[:,i],px[:,i],'.')

