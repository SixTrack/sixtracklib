#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from . import stcommon as st

def  insert_end_of_turn_beam_monitors( elements, num_turn_by_turn_turns,
        turn_by_turn_start, target_num_out_turns, skip_nturns, at_index=None ):
    success = False
    eb = elements.cbuffer
    assert( at_index is None or at_index == eb.n_objects )

    output_turn_start = turn_by_turn_start + num_turn_by_turn_turns

    if num_turn_by_turn_turns > 0:
        elements.BeamMonitor( num_stores=num_turn_by_turn_turns,
                              start=turn_by_turn_start,
                              skip=1, is_rolling=0 )

    if target_num_out_turns > output_turn_start:
        num_stores = target_num_out_turns - output_turn_start
        skip = max( int( skip_nturns ), 1 )

        if skip > 1:
            remainder = num_stores % skip
            num_stores /= skip

            if remainder != 0:
                num_stores += 1

        elements.BeamMonitor( num_stores=num_stores, start=output_turn_start,
                              skip=skip, is_rolling=1 )

    return

# end: python/pysixtracklib/beam_monitor.py