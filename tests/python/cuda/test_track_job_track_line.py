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

    # ==========================================================================
    # Perform "regular" tracking over NUM_TURNS

    cmp_track_pb = pyst.Buffer(cbuffer=cmp_pb)
    assert cmp_track_pb.pointer != st_NullBuffer
    assert cmp_track_pb.num_objects == cmp_pb.n_objects

    cmp_eb = pyst.Buffer(cbuffer=eb)
    assert cmp_eb.pointer != st_NullBuffer
    assert cmp_eb.num_objects == num_beam_elements

    _pset_index = st_buffer_size_t(PSET_INDEX)
    cmp_particles = st_Particles_buffer_get_particles(
        cmp_track_pb.pointer, _pset_index)
    assert cmp_particles != st_NullParticles

    _num_turns = ct.c_int64(NUM_TURNS)
    track_status = st_Track_all_particles_until_turn(
        cmp_particles, cmp_eb.pointer, _num_turns)

    assert track_status == st_TRACK_SUCCESS.value

    # =========================================================================
    # Setup CudaTrackJob to perform the same tracking on the same data but
    # line by line wise:

    track_job = pyst.CudaTrackJob(eb, pb, PSET_INDEX)

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
    # Track particles using track job

    chunk_length = 10
    num_chunks = num_beam_elements // chunk_length
    if 0 != num_beam_elements % chunk_length:
        ++num_chunks

    for ii in range(0, NUM_TURNS):
        for jj in range(0, num_chunks + 1):
            be_begin_idx = jj * chunk_length
            be_end_idx = min(be_begin_idx + chunk_length, num_beam_elements)
            finish_turn = bool(be_end_idx >= num_beam_elements)
            track_job.track_line(be_begin_idx, be_end_idx, finish_turn)
            assert track_job.last_track_status_success

    track_job.collect_particles()
    assert track_job.last_status_success

    # ==========================================================================
    # Compare Results from CPU based and CudaTrackJob based line tracking

    tracked_particles = pb.get_object(PSET_INDEX, cls=pyst.Particles)
    cmp_particles = cmp_pb.get_object(PSET_INDEX, cls=pyst.Particles)

    assert pyst.compareParticlesDifference(
        tracked_particles, cmp_particles, abs_treshold=2e-14) == 0

    sys.exit(0)
