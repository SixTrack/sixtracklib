#!/usr/bin/env python
# -*- coding: utf-8 -*-
import test_data_generation

if __name__ == '__main__':
    test_data_generation.generate_testdata(pyst_example='lhc_no_bb')

prrrrr

import pickle
import os

# Conversion from SixTrack is done using sixtracktools
import sixtracktools

# Tracking is done using pysixtrack
import pysixtrack

# All objects are converted to CObjects stored in a CBuffer I/O buffer
from cobjects import CBuffer

# pysixtracklib provides the CObject based beam elements and particle types
import pysixtracklib as pystlib
from pysixtracklib import export_to_cobjects as etc
from pysixtracklib.particles import Particles as IOParticles

# the path to the input and output folders are provided by the local testdata.py
import testdata

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Step 1: convert the input data into CObject files

    pyst_example = 'lhc_no_bb'

    input_folder  = os.path.join( testdata.PATH_TO_TESTDATA_DIR, pyst_example )
    output_folder = os.path.join( testdata.PATH_TO_TESTDATA_DIR, pyst_example )

    st_beam_elem_dump = os.path.join( output_folder, 'beam_elements_sixtrack.bin' )
    st_particles_dump = os.path.join( output_folder, 'particles_dump_sixtrack.bin' )

    six = sixtracktools.SixInput(input_folder)
    line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)

    beam_elem_buffer = CBuffer()
    etc.line2cobject( line, cbuffer=beam_elem_buffer )
    beam_elem_buffer.to_file( st_beam_elem_dump )

    etc.sixdump2cobject( input_folder,
        os.path.join( input_folder, 'dump3.dat' ), st_particles_dump )

    # -------------------------------------------------------------------------
    # Step 2: Dump particle state into an element by element I/O buffer
    #         before tracking happens at each beam element:

    input_particles_buffer = CBuffer.from_file( st_particles_dump )
    assert( input_particles_buffer.n_objects > 0 )

    ebe_particles_buffer = CBuffer()

    input_particles = input_particles_buffer.get_object( IOParticles, 0 )
    npart = input_particles.num_particles

    pystlib_particles = pystlib.particles.makeCopy( input_particles )

    # one particle is used for the fix-point calculation, thus we would need at
    # least two particles for any kind of non-trivial testdata
    track_particles = []

    for jj in range( npart ):
        track_particles.append( pysixtrack.Particles() )
        input_particles.toPySixTrack( track_particles[ jj ], jj )
        track_particles[ jj ].turn = 0 #Override turn in case it's not 0

    for ii, elem in enumerate( line ):
        label, be_type, beam_element = elem
        before = IOParticles( num_particles=npart, cbuffer=ebe_particles_buffer )

        for jj in range( 0, npart ):
            before.fromPySixTrack( track_particles[ jj ], jj )
            beam_element.track( track_particles[ jj ] )

        before.at_element[:] = ii

    last = IOParticles( num_particles=npart, cbuffer=ebe_particles_buffer )

    for jj in range( npart ):
        last.fromPySixTrack( track_particles[ jj ], jj )

    last.at_turn[:] = 1
    last.at_element[:] = 0

    # -------------------------------------------------------------------------
    # Step 3: Write the element by element I/O buffer to the output file
    #         in the output_folder location:

    assert( ( len( line ) + 1 ) == ebe_particles_buffer.n_objects )

    ebe_particle_dump = os.path.join( output_folder, 'particles_dump.bin' )
    ebe_particles_buffer.to_file( ebe_particle_dump )

    # It's not necessary to dump the beam elements again for the elem-by-elem
    # comparison since the line used therein has not been changed. Still, that
    # way the testdata examples have the same structure and files present:
    beam_elem_dump    = os.path.join( output_folder, 'beam_elements.bin'  )
    beam_elem_buffer.to_file( beam_elem_dump )
