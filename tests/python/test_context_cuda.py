#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import sixtracklib as pyst
from sixtracklib import stcommon as st
import sixtracklib_test as testlib

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    ptr_context = st.st_CudaContext_create()
    assert ptr_context != st.st_NullCudaContext

    st.st_CudaContext_delete(ptr_context)
    ptr_context = st.st_NullCudaContext

# end: sixtracklib/tests/python/test_context_cuda.py
