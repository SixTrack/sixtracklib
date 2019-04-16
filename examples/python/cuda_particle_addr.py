#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ctypes as ct
import pysixtracklib as pyst
from pysixtracklib import stcommon as st
import pycuda

import pycuda.gpuarray
import pycuda.driver
import pycuda.autoinit
import numpy as np

if __name__ == '__main__':
    num_particles = 42
    partset = pyst.ParticlesSet()
    particles = partset.Particles(num_particles=num_particles)
    pbuffer = st.st_Buffer_new_mapped_on_cbuffer( pset.cbuffer )

    ctx = st.st_CudaContext_create()
    particles_arg = st.st_CudaArgument_new( ctx )

    st.st_CudaArgument_send_buffer( particles_arg, pbuffer )

    particles_addr = st.st_ParticlesAddr()
    sizeof_particles_addr = ct.sizeof( st.st_ParticlesAddr )
    sizeof_particles_addr = ct.c_uint64( sizeof_particles_addr )

    ptr_particles_addr = st.st_ParticlesAddr_preset( ct.byref( particles_addr )  )
    particles_addr_arg = st.st_CudaArgument_new( ctx )
    st.st_CudaArgument_send_memory(
        particles_addr_arg, ptr_particles_addr, sizeof_particles_addr )

    st.st_Particles_extract_addresses_cuda(
        st.st_CudaArgument_get_arg_buffer( particles_addr_arg ),
        st.st_CudaArgument_get_arg_buffer( particles_arg ) )

    st.st_CudaArgument_receive_memory(
        particles_addr_arg, ptr_particles_addr, sizeof_particles_addr )

    print( "Particle structure data on the device:" )
    print( "num_particles = {0:8d}".format( particles_addr.num_particles ) )
    print( "x     begin at = {0:16x}".format( particles_addr.x ) )
    print( "y     begin at = {0:16x}".format( particles_addr.y ) )
    print( "px    begin at = {0:16x}".format( particles_addr.px ) )
    print( "py    begin at = {0:16x}".format( particles_addr.py ) )
    print( "zeta  begin at = {0:16x}".format( particles_addr.zeta ) )
    print( "delta begin at = {0:16x}".format( particles_addr.delta ) )

    x = pycuda.gpuarray.GPUArray(
            particles_addr.num_particles, float, gpudata=particles_addr.x )

    x = np.array( [ float( ii ) for ii in range( particles_addr.num_particles ) ] )

    st.st_CudaArgument_receive_buffer( particles_arg, pbuffer )

    cmp_particles = pset.cbuffer.get_object( 0, pyst.Particles )

    assert( pyst.compareParticlesDifference(
            cmp_particles, particles, abs_treshold=2e-14 ) == 0 )

    st.st_CudaContext_delete( ctx )
    st.st_CudaArgument_delete( particles_arg )
    st.st_CudaArgument_delete( particles_addr_arg )
    st.st_Buffer_delete( pbuffer )

    sys.exit( 0 )
