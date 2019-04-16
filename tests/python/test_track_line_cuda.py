#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtracklib as pyst
from   pysixtracklib import stcommon as st
import pysixtracklib_test as testlib
import ctypes as ct
from   cobjects import CBuffer

if __name__== '__main__':
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
    eb = CBuffer.fromfile(path_to_beam_elements_data)

    ctx = st.st_CudaContext_create()

    lattice = st.st_Buffer_new_mapped_on_cbuffer( eb )
    initial_particles = pb.get_object( 0, cls=pyst.Particles )
    track_pb  = CBuffer()
    particles = pyst.makeCopy( initial_particles, cbuffer=track_pb )
    pbuffer   = st.st_Buffer_new_mapped_on_cbuffer( track_pb )

    particles_arg = st.st_CudaArgument_new( ctx )
    success = st.st_CudaArgument_send_buffer( particles_arg, pbuffer )
    assert( success )

    success = st.st_CudaArgument_receive_buffer( particles_arg, pbuffer )
    assert( success )

    lattice_arg = st.st_CudaArgument_new( ctx )
    success = st.st_CudaArgument_send_buffer( lattice_arg, lattice )
    assert( success )

    num_beam_elements = eb.n_objects
    line_begin  = ct.c_uint64( 0 )
    line_middle = ct.c_uint64( num_beam_elements // 2 )
    line_end    = ct.c_uint64( num_beam_elements )

    num_blocks = ct.c_uint64( 1 )
    threads_per_block = ct.c_uint64( 1 )

    st.st_Track_particles_line_cuda_on_grid(
        st.st_CudaArgument_get_arg_buffer( particles_arg ),
        st.st_CudaArgument_get_arg_buffer( lattice_arg ),
        line_begin, line_middle, False, num_blocks, threads_per_block )

    success = st.st_CudaArgument_receive_buffer( particles_arg, pbuffer )

    st.st_Track_particles_line_cuda_on_grid(
        st.st_CudaArgument_get_arg_buffer( particles_arg ),
        st.st_CudaArgument_get_arg_buffer( lattice_arg ),
        line_middle, line_end, True, num_blocks, threads_per_block )

    success = st.st_CudaArgument_receive_buffer( particles_arg, pbuffer )
    print( success )
    assert( success )

    cmp_pb = CBuffer()
    cmp_particles = pyst.makeCopy( initial_particles, cbuffer=cmp_pb )
    cmp_pbuffer = st.st_Buffer_new_mapped_on_cbuffer( cmp_pb )

    st.st_Track_all_particles_until_turn(
        st.st_Particles_buffer_get_particles( cmp_pbuffer, 0 ),
            lattice, ct.c_int64( 1 ) )

    assert( pyst.compareParticlesDifference(
            track_pb.get_object( 0, cls=pyst.Particles ),
            cmp_pb.get_object( 0, cls=pyst.Particles ),
            abs_treshold=2e-14 ) == 0 )

    st.st_CudaArgument_delete( particles_arg )
    st.st_CudaArgument_delete( lattice_arg )
    st.st_CudaContext_delete( ctx )
    st.st_Buffer_delete( pbuffer )
    st.st_Buffer_delete( lattice )
    st.st_Buffer_delete( cmp_pbuffer )


    sys.exit( 0 )
