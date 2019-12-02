#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import sixtracklib as st
from sixtracklib.stcommon import st_NODE_UNDEFINED_INDEX
import sixtracklib_test as testlib
from sixtracklib_test.generic_obj import GenericObj

if __name__ == '__main__':
    if not st.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    try:
        num_nodes = st.CudaController.NUM_AVAILABLE_NODES()
    except RuntimeError as e:
        num_nodes = 0

    if num_nodes <= 0:
        print("No CUDA nodes available -> skip test")
        sys.exit(0)

    ctrl = st.CudaController()
    assert ctrl.num_nodes > 0
    assert ctrl.has_selected_node
    assert ctrl.selected_node_index != st_NODE_UNDEFINED_INDEX.value

    print("ctrl arch_id    : {0}".format(ctrl.arch_id))
    if ctrl.arch_str is not None:
        print("ctrl arch_str   : {0}".format(ctrl.arch_str))

    selected_node_id = ctrl.selected_node_id

    assert isinstance(selected_node_id, st.NodeId)
    assert selected_node_id.is_valid

    node_id_str = str(selected_node_id)
    print("selected node id: {0}".format(selected_node_id))

    selected_node_info = ctrl.selected_node_info
    assert isinstance(selected_node_info, st.NodeInfoBase)

    print("selected node_info: ")
    print(selected_node_info)

    sys.exit(0)
