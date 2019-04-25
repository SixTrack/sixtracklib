#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cobjects import CBuffer
import pysixtracklib as pyst
from pysixtracklib import stcommon as st
import pysixtracklib_test as testlib
import ctypes as ct

if __name__ == '__main__':
    if not pyst.supports('cuda'):
        raise SystemExit("cuda support required for this test")

    num_particles = ct.c_uint64(100)
    pbuffer = st.st_Buffer_new(ct.c_uint64(0))
    assert pbuffer != st.st_NullBuffer

    particles = st.st_Particles_new(pbuffer, num_particles)
    assert particles != st.st_NullParticles

    ptr_context = st.st_CudaContext_create()
    assert ptr_context != st.st_NullCudaContext

    particles_arg = st.st_CudaArgument_new(ptr_context)
    assert particles_arg != st.st_NullCudaArgument

    success = st.st_CudaArgument_send_buffer(particles_arg, pbuffer)
    assert success == 0

    success = st.st_CudaArgument_receive_buffer(particles_arg, pbuffer)
    assert success == 0

    st.st_CudaArgument_delete(particles_arg)
    st.st_CudaContext_delete(ptr_context)
    st.st_Buffer_delete(pbuffer)
    particles = st.st_NullParticles

    sys.exit(0)
