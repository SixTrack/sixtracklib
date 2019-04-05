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

    path_to_beam_elements_data = os.path.join(
        path_to_testdir, "beambeam", "beam_elements.bin" )
    assert( os.path.exists( path_to_beam_elements_data ) )

    pb = st.st_Buffer_new_from_file( path_to_particle_data.encode( 'utf-8' ) )
    eb = st.st_Buffer_new_from_file( path_to_beam_elements_data.encode( 'utf-8' ) )

    particles = st.st_Particles_buffer_get_particles( pb, 0 )
    assert( particles != st.st_NullParticles )
    assert( st.st_Particles_get_num_of_particles( particles ) > 0 )

    num_objects  = ct.c_uint64( 0 )
    num_slots    = ct.c_uint64( 0 )
    num_dataptrs = ct.c_uint64( 0 )
    num_garbage  = ct.c_uint64( 0 )

    slot_size = st.st_Buffer_get_slot_size( pb )
    num_elem_by_elem_turns = 10

    ret = st.st_OutputBuffer_calculate_output_buffer_params( eb, particles,
        num_elem_by_elem_turns, ct.byref( num_objects ), ct.byref( num_slots ),
        ct.byref( num_dataptrs ), ct.byref( num_garbage ), slot_size )

    assert( ret == 0 )
    assert( num_objects.value  > 0 )
    assert( num_slots.value    > 0 )
    assert( num_dataptrs.value > 0 )

    output_buffer = CBuffer(
        max_objects=num_objects.value, max_slots=num_slots.value,
        max_pointers=num_dataptrs.value, max_garbage=num_garbage.value )

    output_buffer_base_addr = output_buffer.base
    output_buffer_size      = output_buffer.size
    saved_max_num_objects   = output_buffer.max_objects
    saved_max_num_slots     = output_buffer.max_slots
    saved_max_num_dataptrs  = output_buffer.max_pointers
    saved_max_num_garbage   = output_buffer.max_garbage

    assert( saved_max_num_objects  >= num_objects.value  )
    assert( saved_max_num_slots    >= num_slots.value    )
    assert( saved_max_num_dataptrs >= num_dataptrs.value )
    assert( saved_max_num_garbage  >= num_garbage.value  )

    ptr_output_buffer_data = ct.cast(
            output_buffer.base, ct.POINTER( ct.c_ubyte ) )

    size_output_buffer_data = ct.c_uint64( output_buffer_size )

    mapped_output_buffer = st.st_Buffer_new_on_data(
        ptr_output_buffer_data, size_output_buffer_data )

    assert( mapped_output_buffer != st.st_NullBuffer )
    assert( st.st_Buffer_get_num_of_objects( mapped_output_buffer ) ==
            output_buffer.n_objects )

    elem_by_elem_offset = ct.c_uint64( 0 )
    beam_monitor_offset = ct.c_uint64( 0 )
    min_turn_id = ct.c_int64( -1 )

    ret = st.st_OutputBuffer_prepare( eb, mapped_output_buffer, particles,
        num_elem_by_elem_turns, ct.byref( elem_by_elem_offset ),
            ct.byref( beam_monitor_offset ), ct.byref( min_turn_id ) )

    assert( ret == 0 )
    print( elem_by_elem_offset.value )
    print( beam_monitor_offset.value )
    print( min_turn_id.value )

    assert( elem_by_elem_offset.value == 0 )
    assert( beam_monitor_offset.value >= elem_by_elem_offset.value )
    assert( min_turn_id.value >= 0 )

    assert( output_buffer.base     == output_buffer_base_addr )
    assert( output_buffer.size     == output_buffer_size )
    assert( saved_max_num_objects  == output_buffer.max_objects  )
    assert( saved_max_num_slots    == output_buffer.max_slots    )
    assert( saved_max_num_dataptrs == output_buffer.max_pointers )
    assert( saved_max_num_garbage  == output_buffer.max_garbage  )

    assert( st.st_Buffer_get_num_of_objects( mapped_output_buffer ) ==
            num_objects.value )

    # Cleanup
    st.st_Buffer_delete( pb )
    st.st_Buffer_delete( eb )

    pb = st.st_NullBuffer
    eb = st.st_NullBuffer

    sys.exit( 0 )

