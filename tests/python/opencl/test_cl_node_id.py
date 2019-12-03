#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sixtracklib as st
from sixtracklib.stcommon import \
    st_ClNodeId, st_ClNodeId_p, st_NullClNodeId, \
    st_NODE_ID_STR_FORMAT_NOARCH, st_NODE_ID_STR_FORMAT_ARCHSTR, \
    st_NODE_ID_STR_FORMAT_ARCHID, st_ARCHITECTURE_OPENCL

if __name__ == '__main__':
    node_a = st.ClNodeId(
        node_id_str="opencl:1.0",
        node_id_str_fmt=st_NODE_ID_STR_FORMAT_ARCHSTR.value)

    assert node_a.pointer != st_NullClNodeId
    assert node_a.owns_node
    assert node_a.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_a.platform_id == 1
    assert node_a.device_id == 0

    node_b = st.ClNodeId(platform_id=2, device_id=4)
    assert node_b.pointer != st_NullClNodeId
    assert node_b.owns_node
    assert node_b.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_b.platform_id == 2
    assert node_b.device_id == 4
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_NOARCH.value) == "2.4"
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_ARCHSTR.value) == \
           "opencl:2.4"

    node_b_link = st.ClNodeId(ext_ptr_node_id=node_b.pointer, owns_ptr=False)
    assert node_b_link.pointer != st_NullClNodeId
    assert node_b_link.pointer == node_b.pointer
    assert not node_b_link.owns_node
    assert node_b_link.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_b_link.platform_id == 2
    assert node_b_link.device_id == 4
    assert node_b_link.to_string(format=st_NODE_ID_STR_FORMAT_NOARCH.value) \
            == "2.4"
    assert node_b_link.to_string(format=st_NODE_ID_STR_FORMAT_ARCHSTR.value) \
            == "opencl:2.4"

    assert node_b.set_device_id( 0 )
    assert node_b.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_b.platform_id == 2
    assert node_b.device_id == 0
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_NOARCH.value) == "2.0"
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_ARCHSTR.value) \
            == "opencl:2.0"

    assert node_b_link.pointer != st_NullClNodeId
    assert node_b_link.pointer == node_b.pointer
    assert not node_b_link.owns_node
    assert node_b_link.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_b_link.platform_id == 2
    assert node_b_link.device_id == 0
    assert node_b_link.to_string(format=st_NODE_ID_STR_FORMAT_NOARCH.value) \
            == "2.0"
    assert node_b_link.to_string(format=st_NODE_ID_STR_FORMAT_ARCHSTR.value) \
            == "opencl:2.0"

    del node_b_link
    node_b_link = None

    assert node_b.pointer != st_NullClNodeId
    assert node_b.owns_node
    assert node_b.arch_id == st_ARCHITECTURE_OPENCL.value
    assert node_b.platform_id == 2
    assert node_b.device_id == 0
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_NOARCH.value) == "2.0"
    assert node_b.to_string(format=st_NODE_ID_STR_FORMAT_ARCHSTR.value) == \
           "opencl:2.0"


