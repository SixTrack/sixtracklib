#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sixtracktools
import pysixtrack

from math import pi
from math import sin, cos
import numpy as np

from cobjects import CBuffer, CObject, CField

from .beam_elements import Drift, DriftExact, Multipole, Cavity, XYShift, SRotation
from .beam_elements import BeamBeam4D, BeamBeam6D
from .particles     import Particles as IOParticles

def sixinput2cobject( input_folder, outfile_name ):
    six = sixtracktools.SixInput(input_folder)
    line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)
    beam_elements = CBuffer()
    line2cobject( line, cbuffer=beam_elements )
    beam_elements.to_file( outfile_name )


def sixdump2cobject( input_folder, st_dump_file , outfile_name ):
    # -------------------------------------------------------------------------
    # Dump particles (element by element)

    six = sixtracktools.SixInput(input_folder)
    line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)

    sixdump = sixtracktools.SixDump101( st_dump_file )


    num_iconv = int( len( iconv ) )
    num_belem = int( len( line  ) )
    num_dumps = int( len( sixdump.particles ) )

    assert(   num_iconv >  0 )
    assert(   num_belem >  iconv[ num_iconv - 1 ]  )
    assert(   num_dumps >= num_iconv )
    assert( ( num_dumps %  num_iconv ) == 0 )

    num_particles = int( num_dumps / num_iconv )

    particles_buffer = CBuffer()

    for ii in range( num_iconv ):
        elem_id = iconv[ ii ]
        assert( elem_id < num_belem )

        p  = IOParticles( cbuffer=particles_buffer,
                          num_particles=num_particles )

        assert( p.num_particles == num_particles )
        assert( len( p.q0 ) == num_particles )

        for jj in range( num_particles ):
            kk = num_particles * ii + jj
            assert( kk < num_dumps )
            p.fromPySixTrack(
                pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() ), jj )
            p.state[ jj ] = 1
            p.at_element[ jj ] = elem_id

    particles_buffer.tofile( outfile_name )
