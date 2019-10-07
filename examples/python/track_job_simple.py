#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import sixtracklib as pyst

if __name__ == '__main__':
    elements = pyst.Elements()
    elements.Drift(length=1.2)
    elements.Multipole(knl=[0, 0.001])

    partset = pyst.ParticlesSet()
    particles = partset.Particles(num_particles=100)

    particles.px += range(len(particles.px))
    particles.px *= 1e-2

    # Print enabled architectures; pass any of these values as arch=
    # to the construction of the track job; default == cpu
    print("enabled archs: {0}".format(
        ', '.join(pyst.TrackJob.enabled_archs())))

    # =========================================================================
    # CPU based Track Job:

    job = pyst.TrackJob(elements.cbuffer, partset.cbuffer)

    # Track until every particle is at the begin of turn 5:
    status = job.track_until(5)  # status should be 0 if success, otherwise < 0

    # Track until every particle is at the begin of turn 10:
    status = job.track_until(10)  # status should be 0 if success, otherwise < 0

    # prepare the particles buffer for read-out:
    job.collect()

    # Track next turn using track_line in two steps:
    status = job.track_line(0, 1)  # Track over the drift, status should be 0
    status = job.track_line(1, 2, finish_turn=True)  # finish tracking line

    del job

    # =========================================================================
    # OpenCL based Track Job:

    if "opencl" in set(pyst.TrackJob.enabled_archs()):
        # Print all available nodes:
        pyst.TrackJob.print_nodes("opencl")

        job = pyst.TrackJob(elements.cbuffer, partset.cbuffer,
                            arch="opencl", device_id="0.0")

        # The particles are still at turn 11 and at element 0 after tracking them
        # with the CPU based track-job; continue tracking!

        status = job.track_until(100)  # track until turn 100
        status = job.track_line(0, 1)
        status = job.track_line(1, 2, finish_turn=True)

        job.collect()

        particles_buffer = job.particles_buffer
        lattice_buffer = job.beam_elements_buffer
        output_buffer = job.output_buffer  # Should be None!!!

        del job

    sys.exit(0)
