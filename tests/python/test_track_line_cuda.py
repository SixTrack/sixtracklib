#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtracklib as pyst
from pysixtracklib import stcommon as st
import pysixtracklib_test as testlib
import ctypes as ct
from cobjects import CBuffer

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("support for cuda required for this test")

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
    initial_particles = pb.get_object(0, cls=pyst.Particles)

    track_pb = CBuffer()
    particles = pyst.makeCopy(initial_particles, cbuffer=track_pb)

    eb = CBuffer.fromfile(path_to_beam_elements_data)
    num_beam_elements = eb.n_objects

    ctx = st.st_CudaContext_create()
    assert ctx != st.st_NullCudaContext

    lattice = st.st_Buffer_new_mapped_on_cbuffer(eb)
    assert lattice != st.st_NullBuffer

    lattice_arg = st.st_CudaArgument_new(ctx)
    assert lattice_arg != st.st_NullCudaArgument

    success = st.st_CudaArgument_send_buffer(lattice_arg, lattice)
    assert success == 0
    assert st.st_CudaArgument_uses_cobjects_buffer(lattice_arg)
    assert not st.st_CudaArgument_uses_raw_argument(lattice_arg)

    assert st.st_CudaArgument_get_size(lattice_arg) == \
        st.st_Buffer_get_size(lattice)

    assert st.st_CudaArgument_get_capacity(lattice_arg) == \
        st.st_Buffer_get_capacity(lattice)

    assert st.st_CudaArgument_has_argument_buffer(lattice_arg)
    assert st.st_CudaArgument_requires_argument_buffer(lattice_arg)

    success = st.st_CudaArgument_receive_buffer(lattice_arg, lattice)
    assert success == 0

    pbuffer = st.st_Buffer_new_mapped_on_cbuffer(track_pb)
    assert pbuffer != st.st_NullBuffer

    particles_arg = st.st_CudaArgument_new(ctx)
    assert particles_arg != st.st_NullCudaArgument

    success = st.st_CudaArgument_send_buffer(particles_arg, pbuffer)
    assert success == 0

    success = st.st_CudaArgument_receive_buffer(particles_arg, pbuffer)
    assert success == 0

    line_begin = ct.c_uint64(0)
    line_middle = ct.c_uint64(num_beam_elements // 2)
    line_end = ct.c_uint64(num_beam_elements)

    num_blocks = ct.c_uint64(32)
    threads_per_block = ct.c_uint64(32)

    st.st_Track_particles_line_cuda_on_grid(
        st.st_CudaArgument_get_arg_buffer(particles_arg),
        st.st_CudaArgument_get_arg_buffer(lattice_arg),
        line_begin, line_middle, ct.c_bool(False),
        num_blocks, threads_per_block)

    success = st.st_CudaArgument_receive_buffer(particles_arg, pbuffer)
    assert success == 0

    st.st_Track_particles_line_cuda_on_grid(
        st.st_CudaArgument_get_arg_buffer(particles_arg),
        st.st_CudaArgument_get_arg_buffer(lattice_arg),
        line_middle, line_end, ct.c_bool(True),
        num_blocks, threads_per_block)

    success = st.st_CudaArgument_receive_buffer(particles_arg, pbuffer)
    assert success == 0

    cmp_pb = CBuffer()
    cmp_particles = pyst.makeCopy(initial_particles, cbuffer=cmp_pb)
    cmp_pbuffer = st.st_Buffer_new_mapped_on_cbuffer(cmp_pb)
    assert cmp_pbuffer != st.st_NullBuffer

    st.st_Track_all_particles_until_turn(
        st.st_Particles_buffer_get_particles(cmp_pbuffer, 0),
        lattice, ct.c_int64(1))

    assert pyst.compareParticlesDifference(
        track_pb.get_object(0, cls=pyst.Particles),
        cmp_pb.get_object(0, cls=pyst.Particles),
        abs_treshold=2e-14) == 0

    st.st_CudaArgument_delete(particles_arg)
    st.st_CudaArgument_delete(lattice_arg)
    st.st_CudaContext_delete(ctx)

    st.st_Buffer_delete(pbuffer)
    st.st_Buffer_delete(lattice)
    st.st_Buffer_delete(cmp_pbuffer)

    sys.exit(0)
