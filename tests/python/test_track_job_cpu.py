#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtrack
import pysixtracklib as pyst
import pysixtracklib.stcommon as st
import ctypes as ct
from   cobjects import CBuffer

if  __name__ == '__main__':
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

    num_elem_by_elem_turns = 1

    pb = CBuffer.fromfile( path_to_particle_data )
    eb = CBuffer.fromfile( path_to_beam_elements_data )

    test_particles = pb.get_object( 0, cls=pyst.Particles )

    #

    sys.exit( 0 )

