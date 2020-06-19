#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import sixtracklib as st
from sixtracklib.stcommon import \
    st_ClNodeId, st_ClNodeId_p, st_NullClNodeId, \
    st_ARCHITECTURE_OPENCL, st_NODE_ID_STR_FORMAT_ARCHSTR, \
    st_NODE_ID_STR_FORMAT_NOARCH, st_NODE_ID_STR_FORMAT_ARCHID

if __name__ == '__main__':
    try:
        total_num_nodes = st.ClController.NUM_ALL_NODES()
    except RuntimeError as e:
        total_num_nodes = 0

    try:
        num_available_nodes = st.ClController.NUM_AVAILABLE_NODES()
    except RuntimeError as e:
        num_available_nodes = 0

    if num_available_nodes <= 0:
        if total_num_nodes > 0:
            print( "OpenCL nodes present, but no available for selection!" )
            print( "--> Check SIXTRACKLIB_DEVICES environment variable!" )
        else:
            print( "No OpenCL nodes available, skip test case" )
        sys.exit(0)

    node_ids = st.ClController.GET_AVAILABLE_NODES()
    assert node_ids and len( node_ids ) > 0 and \
        len( node_ids ) <= num_available_nodes

    for node_id in node_ids:
        assert node_id.pointer != st_NullClNodeId
        assert node_id.owns_node
        assert node_id.arch_id == st_ARCHITECTURE_OPENCL.value
        print( f"{node_id}\r\n" )



    node_id_strs = st.ClController.GET_AVAILABLE_NODE_ID_STRS()
    assert node_id_strs and len( node_id_strs ) == len( node_ids )

    for node_id_str in node_id_strs:
        print( f"{node_id_str}\r\n" )

    sys.exit(0)
