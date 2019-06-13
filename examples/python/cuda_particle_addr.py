#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from importlib import util
import ctypes as ct
import pysixtracklib as pyst
from pysixtracklib import stcommon as st
import pdb

pycuda_spec = util.find_spec('pycuda')
numpy_spec = util.find_spec('numpy')

if pycuda_spec is not None:
    import pycuda
    import pycuda.gpuarray
    import pycuda.driver
    import pycuda.autoinit

if numpy_spec is not None:
    import numpy as np

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("Example requires cuda support in pysixtracklib")

    if pycuda_spec is None:
        raise SystemExit("Example requires pycuda installation")

    if numpy_spec is None:
        raise SystemExit("Example requires numpy installation")

    num_particles = 42
    partset = pyst.ParticlesSet()
    particles = partset.Particles(num_particles=num_particles)

    cmp_partset = pyst.ParticlesSet()
    cmp_particles = cmp_partset.Particles(num_particles=num_particles)

    elements = pyst.Elements()
    elements.Drift(length=1.2)
    elements.Multipole(knl=[0, 0.001])

    pdb.set_trace()
    track_job = pyst.CudaTrackJob(elements, partset)

    if not track_job.has_particle_addresses and \
            track_job.can_fetch_particle_addresses:
        track_job.fetch_particle_addresses()

    pset_index = 0
    ptr_particles_addr = track_job.get_particle_addresses(pset_index)
    particles_addr = ptr_particles_addr.contents

    print("Particle structure data on the device:")
    print("num_particles  = {0:8d}".format(particles_addr.num_particles))
    print("x     begin at = {0:16x}".format(particles_addr.x))
    print("y     begin at = {0:16x}".format(particles_addr.y))
    print("px    begin at = {0:16x}".format(particles_addr.px))
    print("py    begin at = {0:16x}".format(particles_addr.py))
    print("zeta  begin at = {0:16x}".format(particles_addr.zeta))
    print("delta begin at = {0:16x}".format(particles_addr.delta))

    pdb.set_trace()

    x = pycuda.gpuarray.GPUArray(
        particles_addr.num_particles, float, gpudata=particles_addr.x)

    x = np.linspace(0.0, float(num_particles - 1), num=num_particles,
                    dtype=np.float64)

    cmp_particles.x = np.linspace(
        0.0, float(num_particles - 1), num=num_particles, dtype=np.float64)

    assert pyst.compareParticlesDifference(
        cmp_particles, particles, abs_treshold=2e-14) != 0

    track_job.collectParticles()

    assert pyst.compareParticlesDifference(
        cmp_particles, particles, abs_treshold=2e-14) == 0
