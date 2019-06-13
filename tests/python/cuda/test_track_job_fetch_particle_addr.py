#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import ctypes as ct
from cobjects import CBuffer
import pysixtracklib as pyst
from pysixtracklib.stcommon import \
    st_Particles_p, st_ParticlesAddr_p, \
    st_Particles_buffer_get_particles, st_Particles_get_num_of_particles, \
    st_NullParticles, st_NullParticlesAddr,\
    st_NullBuffer, st_ARCH_STATUS_SUCCESS, st_buffer_size_t
import pysixtracklib_test as testlib
from pysixtracklib_test.stcommon import \
    st_TestParticlesAddr_are_addresses_consistent_with_particle
import pdb

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    path_to_testdir = testlib.config.PATH_TO_TESTDATA_DIR
    assert path_to_testdir is not None
    assert os.path.exists(path_to_testdir)
    assert os.path.isdir(path_to_testdir)

    path_to_particle_data = os.path.join(
        path_to_testdir, "beambeam", "particles_dump.bin")
    assert os.path.exists(path_to_particle_data)

    path_to_beam_elements_data = os.path.join(
        path_to_testdir, "beambeam", "beam_elements.bin")
    assert os.path.exists(path_to_beam_elements_data)

    pb = CBuffer.fromfile(path_to_particle_data)
    num_particle_sets = pb.n_objects

    # NOTE: we need the beam elements only for setting up the track job
    eb = CBuffer.fromfile(path_to_beam_elements_data)
    assert eb.n_objects > 0

    # =========================================================================
    # Setup CudaTrackJob to extract device particle addresses:

    track_job = pyst.CudaTrackJob(eb, pb)

    assert track_job.arch_str == "cuda"
    assert track_job.requires_collecting
    assert track_job.is_collecting_particles
    assert track_job.has_controller
    assert track_job.has_particles_arg
    assert track_job.has_beam_elements_arg
    assert track_job.has_particles_addr_arg
    assert not track_job.is_in_debug_mode
    assert not track_job.has_output_buffer
    assert not track_job.owns_output_buffer
    assert not track_job.has_beam_monitor_output
    assert not track_job.has_elem_by_elem_output

    assert track_job.num_particle_sets == 1

    # ==========================================================================
    # Verify that we are able & perform fetching of particle addresses

    assert track_job.can_fetch_particle_addresses

    # NOTE: The assumption is that there are no particle addresses available
    #       at the beginning as obtaining them is potentially costly and not
    #       needed by anybody. The check whether they are present is however
    #       very cheap ->

    if not track_job.has_particle_addresses:
        track_job.fetch_particle_addresses()
        assert track_job.last_status_success

    assert track_job.has_particle_addresses

    pb_buffer = pyst.Buffer(cbuffer=pb)
    assert pb_buffer.pointer != st_NullBuffer
    assert pb_buffer.num_objects == num_particle_sets

    slot_size = pb_buffer.slot_size
    assert slot_size > 0
    _slot_size = st_buffer_size_t( slot_size )

    prev_particle_addr = st_NullParticlesAddr
    for ii in range( 0, num_particle_sets ):
        particle_addr = track_job.get_particle_addresses( ii )
        assert particle_addr != st_NullParticlesAddr
        assert particle_addr != prev_particle_addr

        cmp_particles = st_Particles_buffer_get_particles(
            pb_buffer.pointer, st_buffer_size_t( ii ) )

        assert cmp_particles != st_NullParticles
        assert st_Particles_get_num_of_particles( cmp_particles ) == \
               particle_addr.contents.num_particles

        assert st_TestParticlesAddr_are_addresses_consistent_with_particle(
            particle_addr, cmp_particles, _slot_size )

        prev_particle_addr = particle_addr

    sys.exit(0)
