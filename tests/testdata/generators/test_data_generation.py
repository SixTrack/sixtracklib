#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from pysixtracklib.particles import Particles

# the path to the input and output folders are provided by the local testdata.py
import testdata


def generate_testdata(pyst_example, pysixtrack_line_from_pickle=True):

    # -------------------------------------------------------------------------
    # Step 1: convert the input data into CObject files

    input_folder  = os.path.join( testdata.PATH_TO_TESTDATA_DIR, pyst_example )
    output_folder = os.path.join( testdata.PATH_TO_TESTDATA_DIR, pyst_example )

    st_beam_elem_dump = os.path.join( output_folder, 'beam_elements_sixtrack.bin' )
    beam_elem_dump    = os.path.join( output_folder, 'beam_elements.bin' )

    # Dump the unmodified SixTrack machine description to CBuffer data file
    six = sixtracktools.SixInput(input_folder)
    st_line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)
    st_elements = pystlib.Elements.fromline(st_line)
    st_elements.tofile(st_beam_elem_dump)

    # Dump the pysixtrack machine description to CBuffer data file
    if pysixtrack_line_from_pickle:
        with open(os.path.join(input_folder, 'line.pkl'), 'rb') as fid:
            line = pickle.load(fid)
    else:
        line = st_line


    elements = pystlib.Elements.fromline(line)
    elements.tofile(beam_elem_dump)

    # -------------------------------------------------------------------------
    # Step 2: Dump particle state into an element by element I/O buffer
    #         before tracking happens at each beam element:

    # Dump the unmodified SixTrack element-by-element data to CBuffer data file
    st_particles_dump = os.path.join(output_folder, 'particles_dump_sixtrack.bin')
    st_particles = pystlib.ParticlesSet.fromSixDump101(input_folder,
        os.path.join(input_folder, 'dump3.dat'))
    st_particles.tofile(st_particles_dump)


    # Reload from file
    input_particles_buffer = pystlib.CBuffer.fromfile(st_particles_dump)
    assert(input_particles_buffer.n_objects > 0)

    ebe_particles_buffer = CBuffer()

    input_particles = input_particles_buffer.get_object(0, cls=Particles)
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
        before = Particles( num_particles=npart, cbuffer=ebe_particles_buffer )

        for jj in range( 0, npart ):
            before.fromPySixTrack( track_particles[ jj ], jj )
            beam_element.track( track_particles[ jj ] )

        before.at_element[:] = ii

    last = Particles( num_particles=npart, cbuffer=ebe_particles_buffer )

    for jj in range( npart ):
        last.fromPySixTrack( track_particles[ jj ], jj )

    last.at_turn[:] = 1
    last.at_element[:] = 0

    # -------------------------------------------------------------------------
    # Step 3: Write the element by element I/O buffer to the output file
    #         in the output_folder location:

    assert( ( len( line ) + 1 ) == ebe_particles_buffer.n_objects )
    particles_dump = os.path.join( output_folder, 'particles_dump.bin' )

    ebe_particles_buffer.tofile( particles_dump )
