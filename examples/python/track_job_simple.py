#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pysixtracklib as pyst

elements=pyst.Elements()
elements.Drift(length=1.2)
elements.Multipole(knl=[0,0.001])

partset=pyst.ParticlesSet()
particles=partset.Particles(num_particles=100)

particles.px+=range(len(particles.px))
particles.px*=1e-2

job = pyst.TrackJob(elements.cbuffer,partset.cbuffer)


pyst.TrackJob.print_nodes('opencl')
jobcl = pyst.TrackJob('opencl','0.0',partset.cbuffer,elements.cbuffer)








