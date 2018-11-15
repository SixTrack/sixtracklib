#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import cobjects
import pysixtrack

import particles as tp
import export_to_cobjects as etc

from cobjects import CBuffer

if  __name__ == '__main__':
    # Load pysixtrack line
    pyst_example = 'bbsimple'
    pyst_path = pysixtrack.__file__
    input_folder = '/'.join(pyst_path.split('/')[:-2]+['examples', pyst_example])
    #input_folder = '/home/mschwinz/git/pysixtrack/examples/' + pyst_example
    with open(input_folder+'/line.pkl', 'rb') as fid:
        line = pickle.load(fid)

    # Load sixtracklib ebe data
    dump_to_be_loaded = "../build/examples/c99/stlib_dump.bin"
    buf = cobjects.CBuffer.from_file(dump_to_be_loaded)
    ebe = []

    particles_buffer = CBuffer()
    beam_elements_buffer = CBuffer()

    etc.line2cobject( line, cbuffer=beam_elements_buffer )

    for iob in range(buf.n_objects):
        ebe.append(buf.get_object(tp.Particles, iob))

    assert( len( line ) + 1 <= len( ebe ) )

    for ii, elem in enumerate(line):
        label, be_type, beam_element = elem

        before = ebe[ii]
        next_particles = ebe[ii+1]
        assert( next_particles.num_particles == before.num_particles )

        out_particles = tp.makeCopy( before, cbuffer=particles_buffer )
        out_particles.at_element[:] = ii

        cmp_particles = tp.makeCopy( next_particles )
        cmp_particles.at_element[:] = ii + 1

        tracked_particles = tp.Particles(
                num_particles=before.num_particles )

        print( "beam element: {0}".format( label ) )

        for  jj in range( 0, before.num_particles ):
            ptest = pysixtrack.Particles()
            out_particles.toPySixTrack( ptest, jj )
            beam_element.track( ptest )
            tracked_particles.fromPySixTrack( ptest, jj )
            tracked_particles.at_element[ jj ] = ii + 1

        if  0 == tp.compareParticlesDifference(
            tracked_particles, cmp_particles, abs_treshold=1e-15 ):
            print( " --> success\r\n" )
        else:
            diff = tp.calcParticlesDifference( tracked_particles, cmp_particles )
            print( "difference tracked_particles - cmp_particles:" )
            print( diff )
            print( "\r\n" )

    beam_elements_buffer.to_file( './' + pyst_example + '_beam_elements.bin' )
    particles_buffer.to_file( './' + pyst_example + '_particles_dump.bin' )








