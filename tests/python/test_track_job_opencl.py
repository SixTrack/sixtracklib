#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtrack
import pysixtracklib as pyst
from   pysixtracklib.beam_elements import *
import pysixtracklib_test as testlib
import pysixtracklib.stcommon as st
import ctypes as ct
from   cobjects import CBuffer

import pdb

if  __name__ == '__main__':

    path_to_testdir = testlib.config.PATH_TO_TESTDATA_DIR
    assert( path_to_testdir is not None )
    assert( os.path.exists( path_to_testdir ) )
    assert( os.path.isdir( path_to_testdir ) )

    path_to_particle_data = os.path.join(
        path_to_testdir, "lhc_no_bb", "particles_dump.bin" )
    assert( os.path.exists( path_to_particle_data ) )

    path_to_beam_elements_data = os.path.join(
        path_to_testdir, "lhc_no_bb", "beam_elements.bin" )
    assert( os.path.exists( path_to_beam_elements_data ) )

    pb = CBuffer.fromfile( path_to_particle_data )

    num_elem_by_elem_turns = 1
    eb = CBuffer.fromfile( path_to_beam_elements_data )

    until_turn_elem_by_elem = 1
    until_turn_turn_by_turn = 5
    until_turn              = 100
    skip_turns              = 10

    # ------------------------------------------------------------------------
    initial_num_beam_elements = eb.n_objects

    num_beam_monitors = append_beam_monitors_to_lattice(
        eb, until_turn_elem_by_elem, until_turn_turn_by_turn,
        until_turn, skip_turns )

    num_beam_elements = eb.n_objects
    assert( num_beam_elements ==
           ( initial_num_beam_elements + num_beam_monitors ) )

    # ------------------------------------------------------------------------
    line = testlib.line_from_beam_elem_buffer_pysixtrack( eb )
    assert( line and len( line ) == num_beam_elements )

    # ------------------------------------------------------------------------
    initial_particles = pb.get_object( 0, cls=pyst.Particles )

    cmp_track_pb  = CBuffer()
    cmp_particles = pyst.makeCopy( initial_particles, cbuffer=cmp_track_pb )

    cmp_output_buffer = testlib.track_particles_pysixtrack( cmp_particles,
        line, until_turn_elem_by_elem, until_turn_turn_by_turn,
        until_turn=until_turn, skip_turns=skip_turns )

    pdb.set_trace()
    track_pb = CBuffer()
    track_particles = pyst.makeCopy( initial_particles, cbuffer=track_pb )

    job = pyst.TrackJob( "opencl", device_id_str="0.0",
        particles_buffer=track_pb, beam_elements_buffer=eb,
        until_turn_elem_by_elem=until_turn_elem_by_elem )

    assert( job.type_str() == 'opencl' )
    assert( job.has_output_buffer() )
    assert( job.num_beam_monitors() > 0 )
    assert( job.has_elem_by_elem_outupt() )
    assert( job.has_beam_monitor_output() )

    status = job.track_elem_by_elem( until_turn_elem_by_elem )
    assert( status == 0 )

    status = job.track( until_turn )
    assert( status == 0 )

    job.collect()

    output_buffer = job.output_buffer
    particles_buffer = job.particles_buffer
    tracked_particles = particles_buffer.get_object( 0, cls=pyst.Particles )

    sys.exit( 0 )

