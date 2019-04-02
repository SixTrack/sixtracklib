#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
import argparse
import pysixtracklib as pyst
from pysixtracklib.stcommon import *
from cobjects import CBuffer

import pdb

if __name__ == '__main__':
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














