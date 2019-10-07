#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import sixtracklib as pyst

elements = pyst.Elements()
elements.Drift(length=1.2)
elements.Multipole(knl=[0, 0.001])

particles = pyst.Particles.from_ref(num_particles=10, p0c=1e9)
particles.px += np.linspace(0, 1e-2, 10)

job = pyst.TrackJob(elements, particles)
status = job.track_until(1)

print(particles.x)
print(particles.px)

if pyst.supports('opencl'):
    jobcl = pyst.TrackJob(elements, particles, device="opencl:0.0")
    status = jobcl.track_until(2)
    jobcl.collect()

print(particles.x)
print(particles.px)
