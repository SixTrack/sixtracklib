#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sixtracktools
import pysixtrack

from math import pi
from math import sin, cos
import numpy as np

from cobjects import CBuffer, CObject, CField

from .beam_elements import Drift, DriftExact, MultiPole, Cavity, XYShift, SRotation
from .beam_elements import BeamBeam4D, BeamBeam6D
from .particles     import Particles as IOParticles

def sixinput2cobject( input_folder, outfile_name ):
    six = sixtracktools.SixInput(input_folder)
    line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)
    beam_elements = CBuffer()
    line2cobject( line, cbuffer=beam_elements )
    beam_elements.to_file( outfile_name )


def line2cobject( line, cbuffer=None ):
    deg2rad = pi / 180.0

    # -------------------------------------------------------------------------
    # Dump beam elements:

    if  cbuffer is None: cbuffer = CBuffer()

    for label, elem_type, elem in line:

        if  elem_type == 'Drift':
            e = Drift( cbuffer=cbuffer, length=elem.length )

        elif elem_type == 'DriftExact':
            e = DriftExact( cbuffer=cbuffer, length=elem.length )

        elif elem_type == 'Multipole':
            e = MultiPole( cbuffer=cbuffer, knl=elem.knl, ksl=elem.ksl,
                           length=elem.length, hxl=elem.hxl, hyl=elem.hyl )

        elif elem_type == 'XYShift':
            e = XYShift( cbuffer=cbuffer, dx=elem.dx, dy=elem.dy )

        elif elem_type == 'SRotation':
            angle_rad = deg2rad * elem.angle
            e = SRotation( cbuffer=cbuffer,
                           cos_z=cos( angle_rad ), sin_z=sin( angle_rad ) )

        elif elem_type == 'Cavity':
            e = Cavity( cbuffer=cbuffer, voltage=elem.voltage,
                        frequency=elem.frequency, lag=elem.lag )

        elif elem_type=='BeamBeam4D':

            for mm in ['q_part', 'N_part', 'sigma_x', 'sigma_y', 'beta_s',
                 'min_sigma_diff', 'Delta_x', 'Delta_y', 'Dpx_sub', 'Dpy_sub', 'enabled']:
                 print('4D: %s, %s'%(mm, repr(getattr(elem,mm))))

            data = elem.tobuffer()
            e = BeamBeam4D( cbuffer=cbuffer, data=data)

        elif elem_type=='BeamBeam6D':

            bb6ddata = pysixtrack.BB6Ddata.BB6D_init(
                elem.q_part, elem.N_part_tot, elem.sigmaz, elem.N_slices, elem.min_sigma_diff, elem.threshold_singular,
                elem.phi, elem.alpha,
                elem.Sig_11_0, elem.Sig_12_0, elem.Sig_13_0,
                elem.Sig_14_0, elem.Sig_22_0, elem.Sig_23_0,
                elem.Sig_24_0, elem.Sig_33_0, elem.Sig_34_0, elem.Sig_44_0,
                elem.delta_x, elem.delta_y,
                elem.x_CO, elem.px_CO, elem.y_CO, elem.py_CO, elem.sigma_CO, elem.delta_CO,
                elem.Dx_sub, elem.Dpx_sub, elem.Dy_sub, elem.Dpy_sub, elem.Dsigma_sub, elem.Ddelta_sub,
                elem.enabled)

            N_slices = bb6ddata.N_slices

            print("sphi=%e"%bb6ddata.parboost.sphi);
            print("calpha=%e"%bb6ddata.parboost.calpha);
            print("S33=%e"%bb6ddata.Sigmas_0_star.Sig_33_0);
            print("N_slices=%d"%N_slices);

            for kk in range( N_slices ):
                print("N_part_per_slice[{0}]={1}".format(
                    kk, bb6ddata.N_part_per_slice[kk]) );

            for kk in range( N_slices ):
                print("x_slices_star[{0}]={1}".format(
                    kk, bb6ddata.x_slices_star[kk]) );

            for kk in range( N_slices ):
                print("y_slices_star[{0}]={1}".format(
                    kk, bb6ddata.y_slices_star[kk]) );

            for kk in range( N_slices ):
                print("sigma_slices_star[{0}]={1}".format(
                    kk, bb6ddata.sigma_slices_star[kk]) );

            print("y_CO=%e"%bb6ddata.y_CO);


            data = bb6ddata.tobuffer()
            e = BeamBeam6D( cbuffer=cbuffer, data=data)

        else:
            print( "Unknown/unhandled element type: {0}".format( elem_type, ) )

    return cbuffer


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
            p.at_element[ jj ] = elem_id

    particles_buffer.to_file( outfile_name )


if  __name__ == '__main__':
    # Test on pysixtrack example
    pyst_path = pysixtrack.__file__
    input_folder = '/'.join(pyst_path.split('/')[:-2]+['examples', 'lhc'])


    sixinput2cobject( input_folder, 'lhc_st_input.bin')
    sixdump2cobject( input_folder, input_folder+'/res/dump3.dat', 'lhc_st_dump.bin')
