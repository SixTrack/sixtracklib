#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtracklib as pyst
import pysixtracklib.stcommon as st
import ctypes as ct
from   cobjects import CBuffer


if  __name__ == '__main__':
    # Load the beambeam testcase particle data dump into a sixtracklib
    # NS(Buffer) instance
    path_to_testdir = pyst.config.PATH_TO_TESTDATA_DIR
    assert( path_to_testdir is not None )
    assert( os.path.exists( path_to_testdir ) )
    assert( os.path.isdir( path_to_testdir ) )

    path_to_particle_data = os.path.join(
        path_to_testdir, "beambeam", "particles_dump.bin" )
    assert( os.path.exists( path_to_particle_data ) )

    pb = st.st_Buffer_new_from_file( path_to_particle_data.encode( 'utf-8' ) )
    assert( pb != st.st_NullBuffer )
    num_particle_sets = st.st_Particles_buffer_get_num_of_particle_blocks( pb )
    assert( num_particle_sets > 0 )
    total_num_particles = st.st_Particles_buffer_get_total_num_of_particles( pb )
    assert( total_num_particles > num_particle_sets )
    particles = st.st_Particles_buffer_get_particles( pb, 0 )
    assert( particles != st.st_NullParticles )
    num_particles = st.st_Particles_get_num_of_particles( particles )
    assert( num_particles > 0 )


    # Load the same data file into a CBuffer instance
    cobj_pb = CBuffer.fromfile( path_to_particle_data )
    assert( cobj_pb.n_objects > 0 )
    assert( cobj_pb.n_objects == num_particle_sets )
    cmp_particles = cobj_pb.get_object( 0, cls=pyst.Particles )
    cmp_num_particles = cmp_particles.num_particles
    assert( cmp_particles.num_particles == num_particles )


    # Provide a pyst.Particles instance for calculating the difference
    diff_buffer = CBuffer
    diff = pyst.Particles( cbuffer=diff_buffer, num_particles=cmp_num_particles )
    assert( num_particles == diff.num_particles )
    ptr_diff = ct.cast( diff_buffer.get_object_addr( 0 ), st.st_Particles_p )
    assert( st.st_Particles_get_num_of_particles( ptr_diff ) == num_particles )


    # Calculate the difference between the particles stored on the NS(Buffer)
    # and cmp_particles object, i.e. the CObjects based representation.
    # the difference should be zero
    ptr_cmp_particles = ct.cast( cobj_pb.get_object_addr( 0 ), st.st_Particles_p )
    assert( ptr_cmp_particles != st.st_NullParticles )
    assert( ptr_cmp_particles != particles );
    assert( st.st_Particles_get_num_of_particles( ptr_cmp_particles ) == num_particles )
    st.st_st_Particles_calculate_difference( particles, ptr_cmp_particles, ptr_diff )


    # Cleanup
    st.st_Buffer_delete( pb )
    pb = st.st_NullBuffer
    particles = st.st_NullParticles

    sys.exit( 0 )

#end: tests/python/test_cbuffer_st_buffer.py
