#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import pysixtracklib as pyst
from pysixtracklib.stcommon import st_NODE_UNDEFINED_INDEX
import pysixtracklib_test as testlib
from pysixtracklib_test.generic_obj import GenericObj

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    ctrl = pyst.CudaController()
    assert ctrl.num_nodes > 0
    assert ctrl.has_selected_node
    assert ctrl.selected_node_index != st_NODE_UNDEFINED_INDEX.value

    print( "ctrl arch_id    : {0}".format( ctrl.arch_id ) )
    if ctrl.arch_str is not None:
        print( "ctrl arch_str   : {0}".format( ctrl.arch_str ) )

    selected_node_id = ctrl.selected_node_id

    assert isinstance( selected_node_id, pyst.NodeId )
    assert selected_node_id.is_valid

    node_id_str = str( selected_node_id )
    print( "selected node id: {0}".format( selected_node_id ) )

    selected_node_info = ctrl.selected_node_info
    assert isinstance( selected_node_info, pyst.NodeInfoBase )

    print( "selected node_info: " )
    print( selected_node_info )

    sys.exit( 0 )
