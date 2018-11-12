#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

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

if  __name__ == '__main__':
    py_example   = 'lhc_no_bb'
    input_folder = os.path.join( testdata.PATH_TO_TESTDATA_DIR, py_example )
    path_to_particle_dump = os.path.join( input_folder, 'particles_dump.bin' )
    path_to_line          = os.path.join( input_folder, 'line.pkl' )

    with open( path_to_line, 'rb' ) as fp:
         line = pickle.load( fp )

    input_particles_buffer = CBuffer.from_file( path_to_particle_dump )

    assert( input_particles_buffer.n_objects > 0 )
    track_particles = input_particles_buffer.get_object( IOParticles, 0 )
    num_particles   = track_particles.num_particles

    num_turns = 1
    particles = []

    for jj in range( num_particles ):
        p = pysixtrack.Particles()
        particles.append( p )
        track_particles.toPySixTrack( particles[ jj ], jj )

    track_particles.at_element[:] = 0

    for nn in range( num_turns ):
        for ii, elem in enumerate( line ):
            label, be_type, beam_element = elem

            for jj in range( num_particles ):
                beam_element.track( particles[ jj ] )
                track_particles.fromPySixTrack( particles[ jj ], jj )

            track_particles.at_element[:] = ii

