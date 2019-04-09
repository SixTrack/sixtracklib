#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from   cobjects import CBuffer
import pysixtracklib as st
from   pysixtracklib.particles import Particles as st_Particles
from   pysixtracklib.beam_elements import BeamElement as st_BeamElement
import pysixtrack as pysix
from   pysixtrack.particles import Particles as pysix_Particle

from   pysixtrack.track import Drift, DriftExact, Multipole, XYShift, \
       SRotation, Cavity, RFMultipole, BeamBeam4D, BeamBeam6D, BeamMonitor

def line_from_beam_elem_buffer_pysixtrack( beam_elements_buffer,
        skip_unknown=False ):
    line = []
    num_elements = beam_elements_buffer.n_objects
    for ii in range( num_elements ):
        type_id = beam_elements_buffer.get_object_typeid( ii )
        cls = st_BeamElement.get_elemtype( type_id )
        if  cls is not None:
            line.append( beam_elements_buffer.get_object( ii, cls ) )
        elif skip_unknown is False:
            line.clear()
            break

    return line


def track_particle_pysixtrack( particle, line, until_turn_elem_by_elem,
    until_turn_turn_by_turn, until_turn=0, skip_turns=1):
    output_elem_by_elem = []
    output_turn_by_turn = []
    output_turns = []

    while( particle.turn < until_turn_elem_by_elem ):
        assert( particle.elemid == 0 )
        for elem in line:
            output_elem_by_elem.append( particle.copy() )
            elem.track( particle )
            particle.elemid += 1
        particle.turn  += 1
        particle.elemid = 0

    while( particle.turn < until_turn_turn_by_turn ):
        assert( particle.elemid == 0 )
        for elem in line:
            elem.track( particle )
            particle.elemid += 1
        output_turn_by_turn.append( particle.copy() )
        particle.turn += 1
        particle.elemid = 0

    out_turns_tracked = 0
    while( particle.turn < until_turn ):
        assert( particle.elemid == 0 )
        for elem in line:
            elem.track( particle )
            particle.elemid += 1
        if skip_turns <= 0 or ( out_turns_tracked % skip_turns ) == 0:
            output_turns.append( particle.copy() )

        out_turns_tracked += 1
        particle.turn += 1
        particle.elemid = 0

    return ( output_elem_by_elem, output_turn_by_turn, output_turns )



def track_particles_pysixtrack( input_particles, line, until_turn_elem_by_elem,
    until_turn_turn_by_turn, until_turn=0, skip_turns=1, output_buffer=None ):

    num_particles = input_particles.num_particles
    num_beam_elements = len( line )

    initial_at_turn = sys.maxint
    min_particle_id = sys.maxint
    max_particle_id = -min_particle_id

    for ii in range( num_particles ):
        if input_particles.at_turn[ ii ] < initial_at_turn:
            initial_at_turn = input_particles.at_turn[ ii ]
        if min_particle_id  > input_particles.particle_id[ ii ]:
            min_particle_id = input_particles.particle_id[ ii ]
        if max_particle_id  < input_particles.particle_id[ ii ]:
            max_particle_id = input_particles.particle_id[ ii ]

    nn = ( max_particle_id - min_particle_id + 1 )
    num_particles_per_turn = nn * num_beam_elements

    if output_buffer is None:
        output_buffer = CBuffer()

    if initial_at_turn < until_turn_elem_by_elem:
        num_particles_to_store = num_particles_per_turn * (
            until_turn_elem_by_elem - initial_at_turn )

        out = st_Particles( num_particles_to_store, cbuffer=output_buffer )

        for ii in range( num_particles ):
            particle = pysix_Particle()
            input_particles.to_pysixtrack( particle, ii )

            assert( particle.partid >= min_particle_id )
            assert( particle.partid <= max_particle_id )
            assert( particle.turn >= initial_at_turn )
            assert( particle.turn <  until_turn_elem_by_elem )
            delta_part_id = particle.partid - min_particle_id

            while particle.at_turn < until_turn_elem_by_elem:
                assert( particle.elemid == 0 )
                offset = num_beam_elements * nn * (
                        particle.turn - initial_at_turn )
                for elem in line:
                    out.from_pysixtrack( out,
                        offset + particle.elemid * nn + delta_part_id )
                    elem.track( particle )
                    particle.elemid += 1
                particle.turn += 1
                particle.elemid = 0
            input_particles.from_pysixtrack( particle, ii )

    start_turn_by_turn = max( until_turn_elem_by_elem, initial_at_turn )

    if start_turn_by_turn < until_turn_turn_by_turn:
        max_num_turns_to_store = until_turn_turn_by_turn - start_turn_by_turn
        num_particles_to_store = nn * max_num_turns_to_store
        out = st_Particles( num_particles_to_store, cbuffer=output_buffer )

        for ii in range( num_particles ):
            particle = pysix_Particle()
            input_particles.to_pysixtrack( particle, ii )

            assert( particle.partid >= min_particle_id )
            assert( particle.partid <= max_particle_id )
            assert( particle.turn >= initial_at_turn )
            assert( particle.turn <  until_turn_elem_by_elem )
            delta_part_id = particle.partid - min_particle_id
            turns_tracked = particle.turn - start_turn_by_turn

            while particle.turn < until_turn_turn_by_turn:
                assert( particle.elemid == 0 )
                for elem in line:
                    elem.track( particle )
                    particle.elemid += 1

                out.from_pysixtrack( out, nn * turns_tracked + delta_part_id )
                turns_tracked += 1
                particle.turn += 1
                particle.elemid = 0

            input_particles.from_pysixtrack( particle, ii )

    start_out_turns_turn = max( start_turn_by_turn, until_turn_turn_by_turn )

    if start_out_turns_turn < until_turn:
        max_num_turns_to_store = until_turn - start_out_turns_turn

        if skip_turns <= 0:
            skip_turns = 1

        remainder = max_num_turns_to_store % skip_turns
        if  remainder != 0:
            max_num_turns_to_store += ( skip_turns - remainder )
            assert( max_num_turns_to_store % skip_turns == 0 )

        max_num_turns_to_store /= skip_turns

        num_particles_to_store = nn * max_num_turns_to_store
        out = st_Particles( num_particles_to_store, cbuffer=output_buffer )

        for ii in range( num_particles ):
            particle = pysix_Particle()
            input_particles.to_pysixtrack( particle, ii )

            assert( particle.partid >= min_particle_id )
            assert( particle.partid <= max_particle_id )
            assert( particle.turn >= initial_at_turn )
            assert( particle.turn <  until_turn_elem_by_elem )
            delta_part_id = particle.partid - min_particle_id
            turns_tracked = particle.turn - start_turn_by_turn

            while particle.at_turn < until_turn:
                assert( particle.elemid == 0 )
                for elem in line:
                    elem.track( particle )
                    particle.elemid += 1

                if turns_tracked % skip_turns == 0:
                   jj = nn * ( turns_tracked / skip_turns ) + delta_part_id
                   if  jj > out.num_particles:
                       jj = jj % out.num_particles
                   out.from_pysixtrack( particle, jj )

                turns_tracked  += 1
                particle.turn  += 1
                particle.elemid = 0

            input_particles.from_pysixtrack( particle, ii )

    return output_buffer

# end: sixtracklib/python/pysixtracklib_test