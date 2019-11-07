#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import ctypes as ct
from cobjects import CBuffer
import sixtracklib as pyst
from sixtracklib.stcommon import \
    st_Track_all_particles_until_turn, st_Particles_buffer_get_particles, \
    st_NullParticles, st_NullBuffer, st_buffer_size_t, st_TRACK_SUCCESS
import sixtracklib_test as testlib

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
    cmp_pb = CBuffer.fromfile(path_to_particle_data)

    eb = CBuffer.fromfile(path_to_beam_elements_data)
    num_beam_elements = eb.n_objects

    NUM_TURNS = 10
    PSET_INDEX = 0

    # =========================================================================
    # Setup CudaTrackJob to perform the same tracking on the same data but
    # line by line wise:

    default_threads_per_block = 256
    default_track_threads_per_block = 512

    config_str = f"""
    cuda.default_threads_per_block = {default_threads_per_block}\r\n
    cuda.default_track_threads_per_block = {default_track_threads_per_block}\r\n
    """
    track_job = pyst.CudaTrackJob(eb, pb, PSET_INDEX, config_str=config_str )

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
    assert track_job.default_track_threads_per_block == \
           default_track_threads_per_block
    assert track_job.default_threads_per_block == \
            default_threads_per_block

    sys.exit(0)
