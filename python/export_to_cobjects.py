#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sixtracktools
import pysixtrack
from cobjects import CBuffer, CObject, CField

from beam_elements import Drift, DriftExact, MultiPole, Cavity, XYShift, SRotation
from beam_elements import BeamBeam4D, BeamBeam6D
from particles     import Particles as IOParticles

if  __name__ == '__main__':
    from math import pi
    from math import sin, cos
    import numpy as np
    import pdb

    deg2rad = pi / 180.0

    # -------------------------------------------------------------------------
    # Dump beam elements:

    six = sixtracktools.SixInput('.')
    line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)

    beam_elements = CBuffer()

    for label, elem_type, elem in line:

        if  elem_type == 'Drift':
            e = Drift( cbuffer=beam_elements, length=elem.length )

        elif elem_type == 'DriftExact':
            e = DriftExact( cbuffer=beam_elements, length=elem.length )

        elif elem_type == 'Multipole':
            e = MultiPole( cbuffer=beam_elements, knl=elem.knl, ksl=elem.ksl,
                           length=elem.length, hxl=elem.hxl, hyl=elem.hyl )

        elif elem_type == 'XYShift':
            e = XYShift( cbuffer=beam_elements, dx=elem.dx, dy=elem.dy )

        elif elem_type == 'SRotation':
            angle_rad = deg2rad * elem.angle
            e = SRotation( cbuffer=beam_elements,
                           cos_z=cos( angle_rad ), sin_z=sin( angle_rad ) )

        elif elem_type == 'Cavity':
            e = Cavity( cbuffer=beam_elements, voltage=elem.voltage,
                        frequency=elem.frequency, lag=elem.lag )

        else:
            print( "Unknown/unhandled element type: {0}".format( elem_type, ) )
            pdb.set_trace()

    beam_elements.to_file( './lhc_beam_elements.bin' )

    # -------------------------------------------------------------------------
    # Dump particles (element by element)

    sixdump = sixtracktools.SixDump101('res/dump3.dat')

    num_iconv = int( len( iconv ) )
    num_belem = int( len( line  ) )
    num_dumps = int( len( sixdump ) )

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

            inp = pysixtrack.Particles( **sixdump[ kk ].get_minimal_beam() )
            p.q0[ jj ]         = inp.q0
            p.mass0[ jj ]      = inp.mass0
            p.beta0[ jj ]      = inp.beta0
            p.gamma0[ jj ]     = inp.gamma0
            p.p0c[ jj ]        = inp.p0c
            p.s[ jj ]          = inp.s
            p.x[ jj ]          = inp.x
            p.y[ jj ]          = inp.y
            p.px[ jj ]         = inp.px
            p.py[ jj ]         = inp.py
            p.zeta[ jj ]       = inp.zeta
            p.psigma[ jj ]     = inp.psigma
            p.delta[ jj ]      = inp.delta
            p.rpp[ jj ]        = inp.rpp
            p.rvv[ jj ]        = inp.rvv
            p.chi[ jj ]        = inp.chi
            p.particle[ jj ]   = inp.partid is not None and inp.partid or jj
            p.at_element[ jj ] = elem_id
            p.at_turn[ jj ]    = 0
            p.state[ jj ]      = inp.state

    particles_buffer.to_file( './lhc_particle_dump.bin' )



