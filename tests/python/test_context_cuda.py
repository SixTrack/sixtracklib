#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pysixtracklib as pyst
from   pysixtracklib import stcommon as st
import pysixtracklib_test as testlib

if __name__== '__main__':
    ptr_context = st.st_CudaContext_create()
    assert( ptr_context != st.st_NullCudaContext )

    st.st_CudaContext_delete( ptr_context )
    ptr_context = st.st_NullCudaContext

#end: sixtracklib/tests/python/test_context_cuda.py
