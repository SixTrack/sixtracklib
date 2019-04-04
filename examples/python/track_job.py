#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
import argparse
import pysixtracklib as pyst
from pysixtracklib.stcommon import *
from cobjects import CBuffer

import pdb

if __name__ == '__main__':

    # ========================================================================
    # Setup command line options and arguments

    parser = argparse.ArgumentParser()

    parser.add_argument( "architecture", help="Architecture to be used" )

    parser.add_argument( "-d", "--device", help="Device id string", default="",
                         dest="device_id_str", required=False, )

    parser.add_argument( "-p", "--particles",
                         help="Path to particles dump file",
                         dest="particles_buffer", required=True, )

    parser.add_argument( "-e", "--beam-elements",
                         help="Path to beam-elements dump file",
                         dest="beam_elements_buffer", required=True, )

    parser.add_argument( "-E", "--elem_by_elem_out_turns",
                        help="Number of turns to dump element by element wise",
                        required=False, default=0, type=int,
                        dest="elem_by_elem_out_turns" )

    parser.add_argument( "-T", "--turn_by_turn_out_turns",
                        help="Number of turn-by-turn outputs",
                        required=False, default=0, type=int,
                        dest="turn_by_turn_out_turns" )

    parser.add_argument( "-t", "--target_num_out_turns",
                         help="Target number of end-of-turn outputs",
                         required=False, default=0, type=int,
                         dest="target_num_out_turns" )

    parser.add_argument( "-s", "--skip_out_turns",
                         help="Number of turns between two consequetive outputs " + \
                              "at the end of each turn", type=int,
                         required=False, default=1, dest="skip_out_turns" )

    args = parser.parse_args()

    # ========================================================================
    # Setup input buffers and track job

    path_to_pb = args.particles_buffer
    pb = CBuffer.fromfile( path_to_pb )

    path_to_eb = args.beam_elements_buffer
    beam_elements = pyst.Elements.fromfile( path_to_eb )

    if  args.turn_by_turn_out_turns > 0 or args.target_num_out_turns > 0:
        pyst.insert_end_of_turn_beam_monitors( beam_elements,
            args.turn_by_turn_out_turns, args.elem_by_elem_out_turns,
            args.target_num_out_turns, args.skip_out_turns )

    pdb.set_trace()

    job = pyst.TrackJob( args.architecture, args.device_id_str,
        pb, beam_elements.cbuffer, None, args.elem_by_elem_out_turns )

    # ========================================================================
    # Print summary about configuration

    print( "TrackJob:" )
    print( "----------------------------------------------------------------" )
    print( "architecture            : {0}".format( job.type_str() ) )

    if job.has_elem_by_elem_outupt():
        print( "has elem_by_elem output : yes" )
    else:
        print( "has elem_by_elem output : no" )


    if job.has_beam_monitor_output():
        print( "has beam monitor output : yes" )
        print( "num beam monitors       : {0}".format(
            job.num_beam_monitors() ) )
    else:
        print( "has beam monitor output : no" )

    if  job.has_output_buffer():
        print( "has output buffer       : yes" )
    else:
        print( "has output buffer       : no" )

    print( "num elem by elem turns  : {0:6d}".format( args.elem_by_elem_out_turns ) )
    print( "num turn-by-turn turns  : {0:6d}".format( args.turn_by_turn_out_turns ) )
    print( "target num of turns     : {0:6d}".format( args.target_num_out_turns ) )
    print( "num of turns out skip   : {0:6d}".format( args.skip_out_turns ) )

    # ========================================================================
    # Perform tracking

    success = True
    if args.elem_by_elem_out_turns > 0:
        print( "\r\ntracking {0} turns element by element ... ".format(
            args.elem_by_elem_out_turns ) )

        status  = job.track_elem_by_elem_output( args.elem_by_elem_out_turns )
        success = bool( status == 0 )

        print( "{0}".format( success and "SUCCESS" or "FAILURE" ) )

    if success and args.target_num_out_turns > args.elem_by_elem_out_turns:
        print( "\r\ntracking {0} turns ... ".format(
                args.target_num_out_turns ) )

        status = job.track( args.target_num_out_turns )
        print( status )
        success = bool( status == 0 )

        print( "{0}".format( success and "SUCCESS" or "FAILURE" ) )

    # ========================================================================
    # Collect data before accessing particle_buffer and output_buffer

    if success:
        job.collect()

    # ========================================================================
    # Access the output buffer

    output_buffer = None

    if success and job.has_output_buffer():
        output_buffer = job.output_buffer

        if job.has_elem_by_elem_output():
            assert( output_buffer.n_objects > job.elem_by_elem_output_offset() )
            # These are the particles containing the elem by elem information
            elem_by_elem_particles = output_buffer.get_object(
                 job.elem_by_elem_output_offset() )


        if job.has_beam_monitor_output():
            out_offset = job.beam_monitor_output_offset()
            num_monitors = job.num_beam_monitors()
            for ii in range( out_offset, num_monitors + out_offset ):
                assert( ii < output_buffer.n_objects )
                out_particles = output_buffer.get_object( ii )

    # finished

