import sys
import os
import numpy as np
import sixtracklib as pyst
import sixtracklib_test as testlib

from sixtracklib.stcommon import \
    st_Buffer_new_mapped_on_cbuffer, st_Buffer_delete, \
    st_OutputBuffer_calculate_output_buffer_params, \
    st_OutputBuffer_prepare, st_Track_all_particles_until_turn, \
    st_OutputBuffer_create_output_cbuffer, \
    st_BeamMonitor_assign_output_cbuffer, \
    st_Track_all_particles_element_by_element_until_turn, st_NullBuffer, \
    st_BeamMonitor_assign_output_buffer, st_Buffer_new_mapped_on_cbuffer, \
    st_Particles_cbuffer_get_particles, st_Particles_buffer_get_particles, \
    st_NullParticles, \
    st_Buffer_new_from_copy, st_Buffer_new_mapped_on_cbuffer, \
    st_Buffer_get_num_of_objects, st_Particles_copy

from sixtracklib_test.stcommon import st_Particles_print_out, \
    st_Particles_compare_values_with_treshold, st_Particles_compare_values, \
    st_Particles_buffers_compare_values_with_treshold, \
    st_Particles_random_init

import ctypes as ct
from cobjects import CBuffer

if __name__ == '__main__':
    EPS = np.finfo(float).eps

    line = pyst.Elements()
    line.Drift(length=1.0)
    line.Cavity(voltage=100e3, frequency=400e6, lag=0.0)
    line.Drift(length=3.0)
    line.LimitRect(min_x=-1.0, max_x=1.0, min_y=-2.0, max_y=2.0)
    line.Drift(length=5.0)
    line.BeamMonitor(num_stores=10, start=0, skip=0, out_address=0,
                     max_particle_id=0, min_particle_id=0, is_rolling=False,
                     is_turn_ordered=False)
    line.Drift(length=7.0)
    line.LimitEllipse(a=0.5, b=0.35)
    line.BeamMonitor(num_stores=5, start=10, skip=5, out_address=0,
                     max_particle_id=0, min_particle_id=0, is_rolling=True,
                     is_turn_ordered=False)

    NUM_PARTICLES = 100
    pb = pyst.ParticlesSet()
    pb.Particles(num_particles=100)

    ptr_pb = st_Buffer_new_mapped_on_cbuffer(pb.cbuffer)
    ptr_particles = st_Particles_cbuffer_get_particles(pb.cbuffer, 0)
    assert ptr_particles != st_NullParticles

    job = pyst.CudaTrackJob(line, pb)
    assert job.arch_str == 'cuda'
    assert job.requires_collecting
    assert job.has_output_buffer
    assert not job.has_elem_by_elem_output
    assert job.has_beam_monitor_output
    assert job.num_beam_monitors == 2

    # Copy the original contents of the line, the particle buffer and the out-
    # put buffer to a set of different buffers -> so we can keep track of them

    ptr_pb = st_Buffer_new_mapped_on_cbuffer(pb.cbuffer)
    assert ptr_pb != st_NullBuffer

    ptr_line = st_Buffer_new_mapped_on_cbuffer(line.cbuffer)
    assert ptr_line != st_NullBuffer

    ptr_output = st_Buffer_new_mapped_on_cbuffer(job.output.cbuffer)
    assert ptr_output != st_NullBuffer

    init_pb = st_Buffer_new_from_copy(ptr_pb)
    assert init_pb != st_NullBuffer
    assert st_Buffer_get_num_of_objects(init_pb) == pb.cbuffer.n_objects

    ptr_init_particles = st_Particles_buffer_get_particles(
        init_pb, ct.c_uint64(0))
    assert ptr_init_particles != st_NullParticles
    assert 0 == st_Particles_compare_values(ptr_particles, ptr_init_particles)

    init_line = st_Buffer_new_from_copy(ptr_line)
    assert init_line != st_NullBuffer
    assert st_Buffer_get_num_of_objects(init_line) == \
        line.cbuffer.n_objects

    init_output = st_Buffer_new_from_copy(ptr_output)
    assert init_output != st_NullBuffer
    assert st_Buffer_get_num_of_objects(init_output) == \
        job.output.cbuffer.n_objects

    # Locally change the contents of the line, the particle buffer and the
    # output buffer after the job has been created -> we will push these
    # to the device; also, verify that the local changes are different from
    # the stored-away copies!

    st_Particles_random_init(ptr_particles)
    assert st_Particles_compare_values(ptr_particles, ptr_init_particles) != 0

    for ii in range(0, job.num_beam_monitors):
        ptr_be_mon_output = st_Particles_cbuffer_get_particles(
            job.output.cbuffer, ii)
        ptr_init_be_mon_output = st_Particles_buffer_get_particles(
            init_output, ct.c_uint64(ii))
        assert st_Particles_compare_values(ptr_init_be_mon_output,
                                           ptr_be_mon_output) == 0

        st_Particles_random_init(ptr_be_mon_output)

        assert st_Particles_compare_values(ptr_init_be_mon_output,
                                           ptr_be_mon_output) != 0

    drift1 = line.cbuffer.get_object(0, cls=pyst.Drift)
    assert abs(drift1.length - 1.0) < EPS
    drift1.length *= 0.5

    cavity2 = line.cbuffer.get_object(1, cls=pyst.Cavity)
    assert abs(cavity2.voltage - 100e3) < EPS
    assert abs(cavity2.frequency - 400e6) < EPS
    assert abs(cavity2.lag - 0.0) < EPS
    cavity2.voltage *= 1.10
    cavity2.lag = 2.0

    drift3 = line.cbuffer.get_object(2, cls=pyst.Drift)
    assert abs(drift3.length - 3.0) < EPS
    drift3.length = 0.0

    limit4 = line.cbuffer.get_object(3, cls=pyst.LimitRect)
    assert abs(limit4.min_x + 1.0) < EPS
    assert abs(limit4.max_x - 1.0) < EPS
    limit4.min_x = -2.0
    limit4.max_x = 0.0

    bemon6 = line.cbuffer.get_object(5, cls=pyst.BeamMonitor)
    assert bemon6.out_address != 42
    bemon6.out_address = 42

    limit8 = line.cbuffer.get_object(7, cls=pyst.LimitEllipse)
    assert abs(limit8.a_squ - 0.5 * 0.5) < EPS
    assert abs(limit8.b_squ - 0.35 * 0.35) < EPS
    limit8.set_half_axes(1.0, 1.0)

    bemon9 = line.cbuffer.get_object(8, cls=pyst.BeamMonitor)
    assert bemon9.out_address != 137
    bemon9.out_address = 137

    # Save these local changes to dedicated buffers to allow comparison

    saved_pb = st_Buffer_new_from_copy(ptr_pb)
    assert saved_pb != st_NullBuffer
    assert st_Buffer_get_num_of_objects(saved_pb) == pb.cbuffer.n_objects
    ptr_saved_particles = st_Particles_buffer_get_particles(
        saved_pb, ct.c_uint64(0))
    assert ptr_saved_particles != st_NullParticles
    assert st_Particles_compare_values(
        ptr_saved_particles, ptr_particles) == 0

    saved_line = st_Buffer_new_from_copy(ptr_line)
    assert saved_line != st_NullBuffer
    assert st_Buffer_get_num_of_objects(saved_line) == \
        line.cbuffer.n_objects

    saved_output = st_Buffer_new_from_copy(ptr_output)
    assert saved_output != st_NullBuffer
    assert st_Buffer_get_num_of_objects(saved_output) == \
        job.output.cbuffer.n_objects

    # Push the local changes to the device:

    job.push_particles()
    job.push_beam_elements()
    job._push_output()

    # Modify the local buffers again  and verify that they are different from
    # the saved contents

    st_Particles_copy(ptr_particles, ptr_init_particles)
    assert st_Particles_compare_values(ptr_particles, ptr_saved_particles) != 0
    assert st_Particles_compare_values(ptr_particles, ptr_init_particles) == 0

    line.cbuffer.get_object(0, cls=pyst.Drift).length = 1.0

    cavity2 = line.cbuffer.get_object(1, cls=pyst.Cavity)
    cavity2_saved_voltage = cavity2.voltage
    cavity2.voltage /= 1.10
    cavity2.lag = 0.0

    line.cbuffer.get_object(2, cls=pyst.Drift).length = 3.0

    limit4 = line.cbuffer.get_object(3, cls=pyst.LimitRect)
    limit4.min_x = -1.0
    limit4.max_x = 1.0

    line.cbuffer.get_object(5, cls=pyst.BeamMonitor).out_address = 0

    line.cbuffer.get_object(
        7, cls=pyst.LimitEllipse).set_half_axes(0.5, 0.35)

    line.cbuffer.get_object(8, cls=pyst.BeamMonitor).out_address = 0

    for ii in range(0, job.num_beam_monitors):
        ptr_be_mon_output = st_Particles_cbuffer_get_particles(
            job.output.cbuffer, ii)
        ptr_init_be_mon_output = st_Particles_buffer_get_particles(
            init_output, ct.c_uint64(ii))
        ptr_saved_be_mon_output = st_Particles_buffer_get_particles(
            saved_output, ct.c_uint64(ii))
        assert st_Particles_compare_values(ptr_init_be_mon_output,
                                           ptr_be_mon_output) != 0
        assert st_Particles_compare_values(ptr_saved_be_mon_output,
                                           ptr_be_mon_output) == 0
        st_Particles_copy(ptr_be_mon_output, ptr_init_be_mon_output)
        assert st_Particles_compare_values(ptr_init_be_mon_output,
                                           ptr_be_mon_output) == 0
        assert st_Particles_compare_values(ptr_saved_be_mon_output,
                                           ptr_be_mon_output) != 0

    # Collect again -> this should overwrite the local changes; verify this
    # by comparing against the saved entities

    job.collect_particles()

    assert st_Particles_compare_values(
        st_Particles_cbuffer_get_particles(job.particles_buffer, 0),
        st_Particles_buffer_get_particles(init_pb, ct.c_uint64(0))) != 0

    assert st_Particles_compare_values(
        st_Particles_cbuffer_get_particles(job.particles_buffer, 0),
        st_Particles_buffer_get_particles(saved_pb, ct.c_uint64(0))) == 0

    job.collect_beam_elements()

    drift1 = line.cbuffer.get_object(0, cls=pyst.Drift)
    assert abs(drift1.length - 0.5) < EPS

    cavity2 = line.cbuffer.get_object(1, cls=pyst.Cavity)
    assert abs(cavity2.voltage - cavity2_saved_voltage) < EPS
    assert abs(cavity2.frequency - 400e6) < EPS
    assert abs(cavity2.lag - 2.0) < EPS

    drift3 = line.cbuffer.get_object(2, cls=pyst.Drift)
    assert abs(drift3.length) < EPS

    limit4 = line.cbuffer.get_object(3, cls=pyst.LimitRect)
    assert abs(limit4.min_x + 2.0) < EPS
    assert abs(limit4.max_x) < EPS

    bemon6 = line.cbuffer.get_object(5, cls=pyst.BeamMonitor)
    assert bemon6.out_address == 42

    limit8 = line.cbuffer.get_object(7, cls=pyst.LimitEllipse)
    assert abs(limit8.a_squ - 1.0) < EPS
    assert abs(limit8.b_squ - 1.0) < EPS

    bemon9 = line.cbuffer.get_object(8, cls=pyst.BeamMonitor)
    assert bemon9.out_address == 137

    job.collect_output()

    for ii in range(0, job.num_beam_monitors):
        ptr_be_mon_output = st_Particles_cbuffer_get_particles(
            job.output.cbuffer, ii)
        ptr_init_be_mon_output = st_Particles_buffer_get_particles(
            init_output, ct.c_uint64(ii))
        ptr_saved_be_mon_output = st_Particles_buffer_get_particles(
            saved_output, ct.c_uint64(ii))
        assert st_Particles_compare_values(ptr_init_be_mon_output,
                                           ptr_be_mon_output) != 0
        assert st_Particles_compare_values(ptr_saved_be_mon_output,
                                           ptr_be_mon_output) == 0

    # Cleanup

    st_Buffer_delete(ptr_pb)
    st_Buffer_delete(ptr_line)
    st_Buffer_delete(ptr_output)

    st_Buffer_delete(init_pb)
    st_Buffer_delete(init_line)
    st_Buffer_delete(init_output)

    st_Buffer_delete(saved_pb)
    st_Buffer_delete(saved_line)
    st_Buffer_delete(saved_output)


# end: /python/opencl/test_track_job_collect_and_push_opencl.py
