import numpy as np
import matplotlib.pyplot as pl
from cpymad.madx import Madx
import pysixtracklib as pyst


mad=Madx()
mad.options.echo=False
mad.call(file="SPS_Q20_thin.seq")
mad.use(sequence='sps')
twiss=mad.twiss()
q1mad=twiss.summary['q1']
q2mad=twiss.summary['q2']

elements=pyst.Elements.from_mad(mad.sequence.sps)
nturns=2**14
elements.BeamMonitor(num_stores=nturns)

npart=10
particles=pyst.Particles.from_ref(npart,p0c=26e6)
particles.x += np.linspace(0,1e-6,npart)
job = pyst.TrackJob(elements, particles,until_turn_elem_by_elem=1)
job.track_elem_by_elem(1)
pl.plot(job.output.particles[0].x[1::10])

npart=10
particles=pyst.Particles.from_ref(npart,p0c=26e6)
particles.x += np.linspace(0,1e-6,npart)
job = pyst.TrackJob(elements, particles)
job.track(nturns)

ff=np.linspace(0,0.5,nturns//2+1)
x=job.output.particles[0].x[1::10]
xf=abs(np.fft.rfft(x))
plot(ff,xf)
q1st=ff[xf.argmax()]
print((q1mad-20)-q1st)

