from cpymad.madx import Madx
import pysixtracklib as pyst

import time
from scipy.constants import e, m_p, c

import numpy as np

p0c = 6 * 1e9 # in eV
Etot = np.sqrt(p0c**2 + (m_p/e)**2 * c**4) * 1e-9 # in GeV

mad = Madx()
mad.options.echo = False

mad.call(file="fodo.madx")
mad.command.beam(particle='proton', energy=str(Etot))
mad.use(sequence="FODO")
mad.twiss()

mad.command.select(flag="makethin", class_="quadrupole", slice='8')
mad.command.select(flag="makethin", class_="sbend", slice='8')
mad.command.makethin(makedipedge=False, style="teapot", sequence="fodo")

mad.twiss()

sis18 = mad.sequence.FODO

elements = pyst.Elements.from_mad(sis18)

def prepare(npart=int(1e2), p0c=p0c, elements=elements, device='cpu'):
    particles = pyst.Particles.from_ref(npart, p0c=p0c)
    particles.x += np.linspace(0, 1e-6, npart)

    job = pyst.TrackJob(elements, particles, device=device)
    return job

class Timer(object):
    def __init__(self):
        self.interval = 0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.t1 = time.time()
        self.interval = self.t1 - self.t0
        return False

def timeit(device='cpu:', nturns=nturns, repeat=1):
    res = 0
    for i in range(repeat):
        job = prepare(device=device)
        with Timer() as t:
            job.track((i + 1) * nturns)
        res = (res * i + t.interval) / (i + 1)

    print ('The job took {:.3f} ms for {} turns (mean of {} loops).'.format(
        res * 1e3, nturns, repeat))

nturns = 2
repeat = 1

print ('CPU:')
timeit(device='cpu:',nturns,repeat)

print ('(Trying) OpenCL enabled multi-core CPU:')
timeit(device="opencl:0.0",nturns,repeat)

print ('(Trying) OpenCL enabled multi-core CPU:')
timeit(device="opencl:1.0",nturns,repeat)

print ('(Trying) OpenCL enabled multi-core CPU:')
timeit(device="opencl:2.0",nturns,repeat)

print ('(Trying) OpenCL enabled multi-core CPU:')
timeit(device="opencl:3.0",nturns,repeat)
