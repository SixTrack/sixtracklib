#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtracklib as pyst
import pysixtracklib_test as testlib

from pysixtracklib.stcommon import \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete, \
    st_OutputBuffer_calculate_output_buffer_params, \
    st_OutputBuffer_prepare, st_Track_all_particles_until_turn, \
    st_OutputBuffer_create_output_cbuffer, \
    st_BeamMonitor_assign_output_cbuffer, \
    st_Track_all_particles_element_by_element_until_turn, \
    st_BeamMonitor_assign_output_buffer, st_Buffer_new_mapped_on_cbuffer, \
    st_Particles_cbuffer_get_particles

from pysixtracklib_test.stcommon import st_Particles_print_out, \
    st_Particles_compare_values_with_treshold,\
    st_Particles_buffers_compare_values_with_treshold

import ctypes as ct
from cobjects import CBuffer

if __name__ == '__main__':

    path_to_testdir = testlib.config.PATH_TO_TESTDATA_DIR
    assert(path_to_testdir is not None)
    assert(os.path.exists(path_to_testdir))
    assert(os.path.isdir(path_to_testdir))

    path_to_particle_data = os.path.join(
        path_to_testdir, "beambeam", "particles_dump.bin")
    assert(os.path.exists(path_to_particle_data))

    path_to_beam_elements_data = os.path.join(
        path_to_testdir, "beambeam", "beam_elements.bin")
    assert(os.path.exists(path_to_beam_elements_data))

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
    assert(num_beam_elements ==
           (initial_num_beam_elements + num_beam_monitors))

    # ------------------------------------------------------------------------
    initial_particles = pb.get_object(0, cls=pyst.Particles)

    cmp_track_pb = CBuffer()
    cmp_particles = pyst.makeCopy(initial_particles, cbuffer=cmp_track_pb)

    cmp_output_buffer, elem_by_elem_offset, output_offset, min_turn_id = \
        st_OutputBuffer_create_output_cbuffer(eb,
                                              cmp_track_pb, until_turn_elem_by_elem=until_turn_elem_by_elem)

    assert(cmp_output_buffer.n_objects == 3)
    assert(elem_by_elem_offset == 0)
    assert(output_offset == 1)
    assert(min_turn_id == 0)

    ret = st_BeamMonitor_assign_output_cbuffer(
        eb, cmp_output_buffer, min_turn_id, until_turn_elem_by_elem)

    ptr_belem_buffer = st_Buffer_new_mapped_on_cbuffer(eb)
    ptr_particles = st_Particles_cbuffer_get_particles(cmp_track_pb, 0)
    ptr_elem_by_elem_output = st_Particles_cbuffer_get_particles(
        cmp_output_buffer, elem_by_elem_offset)

    status = st_Track_all_particles_element_by_element_until_turn(
        ptr_particles, ptr_belem_buffer, ct.c_int64(
            until_turn_elem_by_elem), ptr_elem_by_elem_output)

    assert(status == 0)

    status = st_Track_all_particles_until_turn(
        ptr_particles, ptr_belem_buffer, ct.c_int64(until_turn))

    assert(status == 0)

    st_Buffer_delete(ptr_belem_buffer)
    ptr_belem_buffer = pyst.stcommon.st_NullBuffer

    # -------------------------------------------------------------------------

    track_pb = CBuffer()
    track_particles = pyst.makeCopy(initial_particles, cbuffer=track_pb)

    job = pyst.TrackJob("cpu",
                        particles_buffer=track_pb, beam_elements_buffer=eb,
                        until_turn_elem_by_elem=until_turn_elem_by_elem)

    assert(job.type_str() == 'cpu')
    assert(job.has_output_buffer())
    assert(job.num_beam_monitors() > 0)
    assert(job.has_elem_by_elem_outupt())
    assert(job.has_beam_monitor_output())

    status = job.track_elem_by_elem(until_turn_elem_by_elem)
    assert(status == 0)

    status = job.track(until_turn)
    assert(status == 0)

    job.collect()

    output_buffer = job.output_buffer
    particles_buffer = job.particles_buffer

    assert(output_buffer.n_objects == 3)

    ptr_output_buffer = st_Buffer_new_mapped_on_cbuffer(output_buffer)
    ptr_cmp_output_buffer = st_Buffer_new_mapped_on_cbuffer(cmp_output_buffer)
    ABS_DIFF = ct.c_double(2e-14)

    if(0 != st_Particles_buffers_compare_values_with_treshold(
            ptr_output_buffer, ptr_cmp_output_buffer, ABS_DIFF)):

        nn = output_buffer.n_objects

        for ii in range(nn):
            assert(0 == st_Particles_compare_values_with_treshold(
                st_Particles_cbuffer_get_particles(output_buffer, 0),
                st_Particles_cbuffer_get_particles(cmp_output_buffer, 0),
                ABS_DIFF))

    assert(0 == st_Particles_buffers_compare_values_with_treshold(
        ptr_output_buffer, ptr_cmp_output_buffer, ABS_DIFF))

    st_Buffer_delete(ptr_output_buffer)
    st_Buffer_delete(ptr_cmp_output_buffer)

    ptr_output_buffer = pyst.stcommon.st_NullBuffer
    ptr_cmp_output_buffer = pyst.stcommon.st_NullBuffer

    sys.exit(0)
