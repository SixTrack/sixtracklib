#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import sixtracklib as pyst
import sixtracklib_test as pysixtrl_testlib
import sixtracklib.stcommon as st
import ctypes as ct
from cobjects import CBuffer


if __name__ == '__main__':
    # Load the beambeam testcase particle data dump into a sixtracklib
    # NS(Buffer) instance
    path_to_testdir = pysixtrl_testlib.config.PATH_TO_TESTDATA_DIR
    assert(path_to_testdir is not None)
    assert(os.path.exists(path_to_testdir))
    assert(os.path.isdir(path_to_testdir))

    path_to_particle_data = os.path.join(
        path_to_testdir, "beambeam", "particles_dump.bin")
    assert(os.path.exists(path_to_particle_data))

    pb = st.st_Buffer_new_from_file(path_to_particle_data.encode('utf-8'))
    assert(pb != st.st_NullBuffer)
    num_particle_sets = st.st_Particles_buffer_get_num_of_particle_blocks(pb)
    assert(num_particle_sets > 0)
    total_num_particles = st.st_Particles_buffer_get_total_num_of_particles(pb)
    assert(total_num_particles > num_particle_sets)
    particles = st.st_Particles_buffer_get_particles(pb, 0)
    assert(particles != st.st_NullParticles)
    num_particles = st.st_Particles_get_num_of_particles(particles)
    assert(num_particles > 0)

    # Load the same data file into a CBuffer instance
    cobj_pb = CBuffer.fromfile(path_to_particle_data)
    assert(cobj_pb.n_objects > 0)
    assert(cobj_pb.n_objects == num_particle_sets)
    cmp_particles = cobj_pb.get_object(0, cls=pyst.Particles)
    cmp_num_particles = cmp_particles.num_particles
    assert(cmp_particles.num_particles == num_particles)

    # Provide a buffer for calculating the difference
    diff_buffer = st.st_Buffer_new(0)
    assert(diff_buffer != st.st_NullBuffer)

    diff = st.st_Particles_new(diff_buffer, num_particles)
    assert(diff != st.st_NullParticles)
    assert(num_particles == st.st_Particles_get_num_of_particles(diff))

    # Calculate the difference between the particles stored on the NS(Buffer)
    # and cmp_particles object, i.e. the CObjects based representation.
    # the difference should be zero
    ptr_cmp_particles = st.st_Particles_cbuffer_get_particles(cobj_pb, 0)
    assert(ptr_cmp_particles != st.st_NullParticles)
    assert(ptr_cmp_particles != particles)
    assert(st.st_Particles_get_num_of_particles(
        ptr_cmp_particles) == num_particles)

    ABS_ERR = ct.c_double(1e-14)

    assert(0 == pysixtrl_testlib.stcommon.st_Particles_compare_values(
        ptr_cmp_particles, particles)
        or 0 == pysixtrl_testlib.stcommon.st_Particles_compare_values_with_treshold(
        ptr_cmp_particles, particles, ABS_ERR))

    st.st_Particles_calculate_difference(particles, ptr_cmp_particles, diff)
    pysixtrl_testlib.stcommon.st_Particles_print_out(diff)

    # Cleanup
    ptr_cmp_particles = st.st_NullParticles
    particles = st.st_NullParticles
    diff = st.st_NullParticles

    st.st_Buffer_delete(pb)
    pb = st.st_NullBuffer

    st.st_Buffer_delete(diff_buffer)
    diff_buffer = st.st_NullBuffer

    sys.exit(0)

# end: tests/python/test_cbuffer_st_buffer.py
