#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from importlib import util
import ctypes as ct
import sixtracklib as st

numpy_spec = util.find_spec('numpy')

if numpy_spec is not None:
    import numpy as np


def np_array_from_st_addr(base_addr, count, dtype=np.dtype('<f8')):
    typestr = dtype.str
    assert typestr == '<f8' or typestr == '<i8'
    if typestr == '<f8':
        array_t = ct.c_double * count
    elif typestr == '<i8':
        array_t = ct.c_int64 * count
    return np.frombuffer(array_t.from_address(base_addr), dtype=dtype)


if __name__ == '__main__':
    if not st.supports('opencl'):
        raise SystemExit('Example requires opencl support in SixTrackLib')

    if numpy_spec is None:
        raise SystemExit('Example requires numpy installation')

    beam = st.ParticlesSet()
    p1_orig = beam.Particles(num_particles=16, p0c=6.5e12, q0=1.0)
    p2_orig = beam.Particles(num_particles=256, p0c=1e9, q0=1.0)

    lattice = st.Elements()
    dr1 = lattice.Drift(length=0.5)
    qp1 = lattice.Multipole(knl=[0.0, 0.01])
    dr2 = lattice.Drift(length=1.0)
    qp2 = lattice.Multipole(knl=[0.0, -0.01])
    dr3 = lattice.Drift(length=0.5)

    device_str = "opencl:1.0"
    job = st.TrackJob(lattice, beam, device=device_str)

    if not job.can_fetch_particle_addresses:
        raise SystemExit(f'''
            Example requires job for {device_str} to allow
            fetching particle addresses''')

    if not job.has_particle_addresses and job.can_fetch_particle_addresses:
        job.fetch_particle_addresses()

    ptr_p1_addr = job.get_particle_addresses(0)
    ptr_p2_addr = job.get_particle_addresses(1)

    # Printing the particle addresses
    for label, ptr_p_addr in {'p1': ptr_p1_addr, 'p2': ptr_p2_addr}.items():
        p_addr = ptr_p_addr.contents
        print(f"""particle addresses for particle set {label}:\r\n
          num_particles   = {p_addr.num_particles:8d}\r\n
          x      begin at = {p_addr.x:16x} (alignment {p_addr.x%128 == 0})\r\n
          y      begin at = {p_addr.y:16x} (alignment {p_addr.x%128 == 0})\r\n
          px     begin at = {p_addr.px:16x} (alignment {p_addr.x%128 == 0})\r\n
          py     begin at = {p_addr.py:16x} (alignment {p_addr.x%128 == 0})\r\n
          zeta   begin at = {p_addr.zeta:16x} (alignment {p_addr.x%128 == 0})\r\n
          delta  begin at = {p_addr.delta:16x} (alignment {p_addr.x%128 == 0})\r\n\r\n""")

    # Creating np arrays from the addresses
    p1_addr = ptr_p1_addr.contents
    p1_npart = p1_addr.num_particles
    p1_x = np_array_from_st_addr(p1_addr.x, count=p1_npart)
    p1_px = np_array_from_st_addr(p1_addr.px, count=p1_npart)
    p1_p0c = np_array_from_st_addr(p1_addr.p0c, count=p1_npart)
    p1_state = np_array_from_st_addr(
        p1_addr.state,
        count=p1_npart,
        dtype=np.dtype('<i8'))

    # getting the p1 particle set from the job:
    p1 = job.particles_buffer.get_object(0, cls=st.Particles)

    # compare p1 on the host with the mapped arrays on the device side ->
    # should be equal
    assert np.array_equal(p1.x, p1_x)
    assert np.array_equal(p1.px, p1_px)
    assert np.array_equal(p1.p0c, p1_p0c)
    assert np.array_equal(p1.state, p1_state)

    # Update the content of the device side buffers:
    p1_x[:] = np.random.uniform(-1e-5, +1e-5, p1_npart)
    p1_px[:] = np.random.uniform(-1e-5, +1e-5, p1_npart)
    p1_p0c[:] = np.ones(p1_npart) * 1e9
    p1_state[:] = np.random.randint(0, 2, p1_npart)

    # Verify that the host side p1 and the device side mapped arrays are
    # no longer equal
    assert not np.array_equal(p1.x, p1_x)
    assert not np.array_equal(p1.px, p1_px)
    assert not np.array_equal(p1.p0c, p1_p0c)
    assert not np.array_equal(p1.state, p1_state)

    # collect the particles -> this should update the host-side p1 with the
    # data from the device
    job.collect_particles()

    # Print output:
    print(f"""p1 after the update + collect_particles():\r\n
        p1.x     : {p1.x} (from SixTrackLib) == \r\n
        p1_x     : {p1_x} (mapped address) \r\n
        p1.px    : {p1.px} (from SixTrackLib) \r\n
        p1_px    : {p1_px} (mapped address) \r\n
        p1.p0c   : {p1.p0c} (from SixTrackLib) \r\n
        p1_p0c   : {p1_p0c} (mapped address) \r\n
        p1.state : {p1.state} (from SixTrackLib) \r\n
        p1_state : {p1_state} (mapped address) \r\n""")

    # We expect, that the host side and the mapped device side arrays agree
    # again:
    assert np.array_equal(p1.x, p1_x)
    assert np.array_equal(p1.px, p1_px)
    assert np.array_equal(p1.p0c, p1_p0c)
    assert np.array_equal(p1.state, p1_state)
