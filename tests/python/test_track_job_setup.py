import sys
import os
import sixtracklib as pyst
import sixtracklib_test as testlib

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

    # -------------------------------------------------------------------------
    eb = CBuffer.fromfile(path_to_beam_elements_data)
    eb_data_begin = eb.base
    eb_data_size = eb.size
    eb_num_objects = eb.n_objects

    pb = CBuffer.fromfile(path_to_particle_data)
    assert(pb.n_objects > 0)
    particle_type_id = pb.get_object_typeid(0)

    pb_data_begin = pb.base
    pb_data_size = pb.size
    pb_num_objects = pb.n_objects

    # Testcase 1: only default parameters, no elem by elem output, no
    # beam monitors

    job = pyst.TrackJob(eb, pb)

    assert(job.arch_str == 'cpu')
    assert(job.particles_buffer == pb)
    assert(job.beam_elements_buffer == eb)
    assert(not job.has_output_buffer)
    assert(job.output_buffer is None)
    assert(not job.has_elem_by_elem_output)
    assert(not job.has_beam_monitor_output)
    assert(job.num_beam_monitors == 0)
    assert(job.elem_by_elem_output_offset == 0)
    assert(job.beam_monitor_output_offset == 0)

    del job
    job = None

    assert(eb.base == eb_data_begin)
    assert(eb.size == eb_data_size)
    assert(eb.n_objects == eb_num_objects)

    assert(pb.base == pb_data_begin)
    assert(pb.size == pb_data_size)
    assert(pb.n_objects == pb_num_objects)

    # -------------------------------------------------------------------------
    # Testcase 2: only elem by elem output, no beam monitors

    until_turn_elem_by_elem = 5

    job = pyst.TrackJob(eb, pb, until_turn_elem_by_elem)

    assert(job.arch_str == 'cpu')
    assert(job.particles_buffer == pb)
    assert(job.beam_elements_buffer == eb)
    assert(job.has_output_buffer)
    assert(job.output_buffer is not None)
    assert(job.output_buffer.n_objects == 1)

    assert(job.has_elem_by_elem_output)
    assert(job.elem_by_elem_output_offset < job.output_buffer.n_objects)
    elem_by_elem_offset = job.elem_by_elem_output_offset

    assert(job.output_buffer.get_object_typeid(elem_by_elem_offset) ==
           particle_type_id)

    assert(not job.has_beam_monitor_output)
    assert(job.num_beam_monitors == 0)
    assert(job.elem_by_elem_output_offset == 0)
    assert(job.beam_monitor_output_offset >
           job.elem_by_elem_output_offset)

    del job
    job = None

    assert(eb.base == eb_data_begin)
    assert(eb.size == eb_data_size)
    assert(eb.n_objects == eb_num_objects)

    assert(pb.base == pb_data_begin)
    assert(pb.size == pb_data_size)
    assert(pb.n_objects == pb_num_objects)

    # -------------------------------------------------------------------------
    # Testcase 3: only beam monitors, no elem by elem output

    until_turn_elem_by_elem = 0
    until_turn_turn_by_turn = 5
    until_turn_output = 100
    skip_turns = 10

    num_beam_monitors = pyst.append_beam_monitors_to_lattice(
        eb,
        until_turn_elem_by_elem,
        until_turn_turn_by_turn,
        until_turn_output,
        skip_turns=skip_turns)

    assert(num_beam_monitors == 2)
    assert(eb_num_objects + num_beam_monitors == eb.n_objects)

    eb_num_objects = eb.n_objects
    eb_data_begin = eb.base
    eb_data_size = eb.size

    job = pyst.TrackJob(eb, pb, until_turn_elem_by_elem)

    assert(job.arch_str == 'cpu')
    assert(job.particles_buffer == pb)
    assert(job.beam_elements_buffer == eb)
    assert(job.has_output_buffer)
    assert(job.output_buffer is not None)
    assert(job.output_buffer.n_objects == num_beam_monitors)

    assert(not job.has_elem_by_elem_output)
    assert(job.elem_by_elem_output_offset == 0)
    assert(job.beam_monitor_output_offset == 0)
    beam_monitor_output_offset = job.beam_monitor_output_offset

    assert(job.output_buffer.get_object_typeid(beam_monitor_output_offset) ==
           particle_type_id)

    assert(job.has_beam_monitor_output)
    assert(job.num_beam_monitors == num_beam_monitors)

    del job
    job = None

    assert(eb.base == eb_data_begin)
    assert(eb.size == eb_data_size)
    assert(eb.n_objects == eb_num_objects)

    assert(pb.base == pb_data_begin)
    assert(pb.size == pb_data_size)
    assert(pb.n_objects == pb_num_objects)

    # -------------------------------------------------------------------------
    # Testcase 4: elem by elem output + beam_monitors

    until_turn_elem_by_elem = 2
    until_turn_turn_by_turn = 5
    until_turn_output = 100
    skip_turns = 10

    job = pyst.TrackJob(eb, pb, until_turn_elem_by_elem)

    assert(job.arch_str == 'cpu')
    assert(job.particles_buffer == pb)
    assert(job.beam_elements_buffer == eb)
    assert(job.has_output_buffer)
    assert(job.output_buffer is not None)
    assert(job.output_buffer.n_objects == (num_beam_monitors + 1))

    assert(job.has_elem_by_elem_output)
    assert(job.has_beam_monitor_output)

    elem_by_elem_output_offset = job.elem_by_elem_output_offset
    beam_monitor_output_offset = job.beam_monitor_output_offset
    assert(elem_by_elem_output_offset == 0)
    assert(elem_by_elem_output_offset < beam_monitor_output_offset)
    assert(beam_monitor_output_offset + num_beam_monitors <=
           job.output_buffer.n_objects)

    assert(job.output_buffer.get_object_typeid(
        beam_monitor_output_offset) == particle_type_id)

    for ii in range(num_beam_monitors):
        jj = ii + beam_monitor_output_offset
        assert(job.output_buffer.get_object_typeid(jj) == particle_type_id)

    assert(job.has_beam_monitor_output)
    assert(job.num_beam_monitors == num_beam_monitors)

    del job
    job = None

    assert(eb.base == eb_data_begin)
    assert(eb.size == eb_data_size)
    assert(eb.n_objects == eb_num_objects)

    assert(pb.base == pb_data_begin)
    assert(pb.size == pb_data_size)
    assert(pb.n_objects == pb_num_objects)

    sys.exit(0)

# end: tests/python/test_track_job_setup.py
