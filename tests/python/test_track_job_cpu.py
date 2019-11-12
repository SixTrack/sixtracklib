#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import sixtracklib as pyst
import sixtracklib_test as testlib

from sixtracklib.stcommon import \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete, \
    st_OutputBuffer_calculate_output_buffer_params, \
    st_OutputBuffer_prepare, st_Track_all_particles_until_turn, \
    st_OutputBuffer_create_output_cbuffer, \
    st_ElemByElemConfig, st_ElemByElemConfig_p, st_NullElemByElemConfig, \
    st_ElemByElemConfig_create, st_ElemByElemConfig_delete, \
    st_ElemByElemConfig_init, st_ElemByElemConfig_assign_output_cbuffer, \
    st_BeamMonitor_assign_output_cbuffer, \
    st_Track_all_particles_element_by_element_until_turn, \
    st_Particles_buffer_get_particles, \
    st_BeamMonitor_assign_output_buffer, st_Buffer_new_mapped_on_cbuffer, \
    st_Particles_cbuffer_get_particles, st_NullBuffer, \
    st_ARCH_STATUS_SUCCESS, st_particle_index_t

from sixtracklib_test.stcommon import st_Particles_print_out, \
    st_Particles_compare_values_with_treshold,\
    st_Particles_buffers_compare_values_with_treshold

import ctypes as ct
from cobjects import CBuffer

if __name__ == '__main__':
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

    num_elem_by_elem_turns = 1
    eb = CBuffer.fromfile(path_to_beam_elements_data)

    until_turn_elem_by_elem = 1
    until_turn_turn_by_turn = 5
    until_turn = 100
    skip_turns = 10

    # ------------------------------------------------------------------------
    initial_num_beam_elements = eb.n_objects

    num_beam_monitors = pyst.beam_elements.append_beam_monitors_to_lattice(
        eb, until_turn_elem_by_elem, until_turn_turn_by_turn,
        until_turn, skip_turns)

    num_beam_elements = eb.n_objects
    assert num_beam_elements == (initial_num_beam_elements + num_beam_monitors)

    # ------------------------------------------------------------------------
    initial_particles = pb.get_object(0, cls=pyst.Particles)

    cmp_track_pb = CBuffer()
    cmp_particles = pyst.makeCopy(initial_particles, cbuffer=cmp_track_pb)

    cmp_output_buffer, elem_by_elem_offset, output_offset, min_turn_id = \
        st_OutputBuffer_create_output_cbuffer(eb,
                                              cmp_track_pb,
                                              until_turn_elem_by_elem=until_turn_elem_by_elem)

    elem_by_elem_config = st_ElemByElemConfig_create()
    assert elem_by_elem_config != st_NullElemByElemConfig

    start_elem_idx_arg = st_particle_index_t(0)
    until_turn_elem_by_elem_arg = st_particle_index_t(until_turn_elem_by_elem)

    ptr_belem_buffer = st_Buffer_new_mapped_on_cbuffer(eb)
    ptr_particles = st_Particles_cbuffer_get_particles(cmp_track_pb, 0)

    status = st_ElemByElemConfig_init(
        elem_by_elem_config,
        ptr_particles,
        ptr_belem_buffer,
        start_elem_idx_arg,
        until_turn_elem_by_elem_arg)
    assert status == 0

    assert cmp_output_buffer.n_objects == 3
    assert elem_by_elem_offset == 0
    assert output_offset == 1
    assert min_turn_id == 0

    status = st_ElemByElemConfig_assign_output_cbuffer(
        elem_by_elem_config, cmp_output_buffer, elem_by_elem_offset)

    assert status == 0

    status = st_BeamMonitor_assign_output_cbuffer(
        eb, cmp_output_buffer, min_turn_id, until_turn_elem_by_elem)

    assert status == 0

    status = st_Track_all_particles_element_by_element_until_turn(
        ptr_particles, elem_by_elem_config, ptr_belem_buffer,
        until_turn_elem_by_elem_arg)

    assert status == 0

    status = st_Track_all_particles_until_turn(
        ptr_particles, ptr_belem_buffer, st_particle_index_t(until_turn))

    assert status == 0

    st_Buffer_delete(ptr_belem_buffer)
    st_ElemByElemConfig_delete(elem_by_elem_config)

    ptr_belem_buffer = st_NullBuffer
    elem_by_elem_config = st_NullElemByElemConfig

    # -------------------------------------------------------------------------

    track_pb = CBuffer()
    track_particles = pyst.makeCopy(initial_particles, cbuffer=track_pb)

    job = pyst.TrackJob(eb, track_pb, until_turn_elem_by_elem)

    assert job.arch_str == 'cpu'
    assert job.has_output_buffer
    assert job.num_beam_monitors > 0
    assert job.has_elem_by_elem_output
    assert job.has_beam_monitor_output

    status = job.track_elem_by_elem(until_turn_elem_by_elem)
    assert status == 0

    status = job.track_until(until_turn)
    assert status == 0

    job.collect()

    output_buffer = job.output_buffer
    particles_buffer = job.particles_buffer

    assert output_buffer.size > 0
    assert output_buffer.n_objects > 0

    assert cmp_output_buffer.n_objects == output_buffer.n_objects
    assert cmp_output_buffer.base != output_buffer.base

    nn = cmp_output_buffer.n_objects
    ABS_DIFF = 2e-14

    for ii in range(nn):
        cmp_particles = cmp_output_buffer.get_object(ii, pyst.Particles)
        particles = output_buffer.get_object(ii, pyst.Particles)
        assert(0 == pyst.particles.compareParticlesDifference(
            cmp_particles, particles, abs_treshold=ABS_DIFF))

    del job
    del track_pb
    del cmp_track_pb
    del particles
    del cmp_particles
    del elem_by_elem_config

    # -------------------------------------------------------------------------
    # track line:

    track_pb = CBuffer()
    track_particles = pyst.makeCopy(initial_particles, cbuffer=track_pb)

    cmp_pb = CBuffer()
    cmp_particles = pyst.makeCopy(initial_particles, cbuffer=cmp_pb)
    cmp_pbuffer = st_Buffer_new_mapped_on_cbuffer(cmp_pb)
    assert(cmp_pbuffer != st_NullBuffer)

    lattice = st_Buffer_new_mapped_on_cbuffer(eb)
    assert(lattice != st_NullBuffer)

    until_turn = 10

    st_Track_all_particles_until_turn(st_Particles_buffer_get_particles(
        cmp_pbuffer, 0), lattice, ct.c_int64(until_turn))

    st_Buffer_delete(lattice)
    lattice = st_NullBuffer

    job = pyst.TrackJob(eb, track_pb)

    num_beam_elements = eb.n_objects
    num_lattice_parts = 10
    num_elem_per_part = num_beam_elements // num_lattice_parts

    for ii in range(until_turn):
        for jj in range(num_lattice_parts):
            is_last_in_turn = bool(jj == (num_lattice_parts - 1))
            begin_idx = jj * num_elem_per_part
            end_idx = (not is_last_in_turn) \
                and begin_idx + num_elem_per_part \
                or num_beam_elements

            status = job.track_line(begin_idx, end_idx, is_last_in_turn)

    job.collect()

    particles_buffer = job.particles_buffer
    track_particles = particles_buffer.get_object(0, cls=pyst.Particles)

    assert(0 == pyst.particles.compareParticlesDifference(
        cmp_particles, track_particles, abs_treshold=ABS_DIFF))

    sys.exit(0)
