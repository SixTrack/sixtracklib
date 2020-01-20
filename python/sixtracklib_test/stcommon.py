#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as ct
from . import config as stconf
from sixtracklib.particles import Particles as st_Particles
import sixtracklib.stcommon as stcom

testlib = ct.CDLL(stconf.SHARED_LIB)

# C-API Types

st_Null = stcom.st_Null
st_NullChar = stcom.st_NullChar

st_uint64_p = stcom.st_uint64_p
st_uchar_p = stcom.st_uchar_p

st_double_p = stcom.st_double_p
st_int64_p = stcom.st_int64_p

st_Buffer = stcom.st_Buffer
st_Buffer_p = stcom.st_Buffer_p

# -----------------------------------------------------------------------------
# Particles realted functions and definitions

st_Particles = stcom.st_Particles
st_Particles_p = stcom.st_Particles_p
st_NullParticles = stcom.st_NullParticles


st_Particles_realistic_init = testlib.st_Particles_realistic_init
st_Particles_realistic_init.argtypes = [st_Particles_p]
st_Particles_realistic_init.restype = None


st_Particles_random_init = testlib.st_Particles_random_init
st_Particles_random_init.argtypes = [st_Particles_p]
st_Particles_random_init.restype = None


st_Particles_have_same_structure = testlib.st_Particles_have_same_structure_ext
st_Particles_have_same_structure.argtypes = [st_Particles_p, st_Particles_p]
st_Particles_have_same_structure.restype = ct.c_bool


st_Particles_map_to_same_memory = testlib.st_Particles_map_to_same_memory_ext
st_Particles_map_to_same_memory.argtypes = [st_Particles_p, st_Particles_p]
st_Particles_map_to_same_memory.restype = ct.c_bool


st_Particles_compare_real_values = testlib.st_Particles_compare_real_values_ext
st_Particles_compare_real_values.restype = ct.c_int32
st_Particles_compare_real_values.argtypes = [st_Particles_p, st_Particles_p]


st_Particles_compare_real_values_with_treshold = (
    testlib.st_Particles_compare_real_values_with_treshold_ext
)
st_Particles_compare_real_values_with_treshold.restype = ct.c_int32
st_Particles_compare_real_values_with_treshold.argtypes = [
    st_Particles_p,
    st_Particles_p,
]


st_Particles_compare_integer_values = (
    testlib.st_Particles_compare_integer_values_ext
)
st_Particles_compare_integer_values.restype = ct.c_int32
st_Particles_compare_integer_values.argtypes = [st_Particles_p, st_Particles_p]


st_Particles_compare_values = testlib.st_Particles_compare_values_ext
st_Particles_compare_values.argtypes = [st_Particles_p, st_Particles_p]
st_Particles_compare_values.restypes = ct.c_int32


st_Particles_compare_values_with_treshold = (
    testlib.st_Particles_compare_values_with_treshold_ext
)
st_Particles_compare_values_with_treshold.restype = ct.c_int32
st_Particles_compare_values_with_treshold.argtypes = [
    st_Particles_p,
    st_Particles_p,
    ct.c_double,
]


st_Particles_print_out_single = testlib.st_Particles_print_out_single_ext
st_Particles_print_out_single.restype = None
st_Particles_print_out_single.argtypes = [st_Particles_p, ct.c_uint64]


st_Particles_print_out = testlib.st_Particles_print_out_ext
st_Particles_print_out.restype = None
st_Particles_print_out.argtypes = [st_Particles_p]


st_Particles_buffers_map_to_same_memory = (
    testlib.st_Particles_buffers_map_to_same_memory
)
st_Particles_buffers_map_to_same_memory.restype = ct.c_bool
st_Particles_buffers_map_to_same_memory.argtypes = [st_Buffer_p, st_Buffer_p]


st_Particles_buffer_have_same_structure = (
    testlib.st_Particles_buffer_have_same_structure
)
st_Particles_buffer_have_same_structure.restype = ct.c_bool
st_Particles_buffer_have_same_structure.argtypes = [st_Buffer_p, st_Buffer_p]


st_Particles_buffers_compare_values = (
    testlib.st_Particles_buffers_compare_values
)
st_Particles_buffers_compare_values.restype = ct.c_int32
st_Particles_buffers_compare_values.argtypes = [st_Buffer_p, st_Buffer_p]


st_Particles_buffers_compare_values_with_treshold = (
    testlib.st_Particles_buffers_compare_values_with_treshold
)
st_Particles_buffers_compare_values.restype = ct.c_int32
st_Particles_buffers_compare_values.argtypes = [
    st_Buffer_p,
    st_Buffer_p,
    ct.c_double,
]


st_Particles_buffer_print_out = testlib.st_Particles_buffer_print_out
st_Particles_buffer_print_out.restype = None
st_Particles_buffer_print_out.argtypes = [st_Buffer_p]

# -----------------------------------------------------------------------------
# ParticlesAddr realted functions and definitions

st_ParticlesAddr = stcom.st_ParticlesAddr
st_ParticlesAddr_p = stcom.st_ParticlesAddr_p
st_NullParticlesAddr = stcom.st_NullParticlesAddr

st_buffer_size_t = stcom.st_buffer_size_t

st_TestParticlesAddr_are_addresses_consistent_with_particle = (
    testlib.st_TestParticlesAddr_are_addresses_consistent_with_particle
)
st_TestParticlesAddr_are_addresses_consistent_with_particle.argtypes = [
    st_ParticlesAddr_p,
    st_Particles_p,
    st_buffer_size_t,
]
st_TestParticlesAddr_are_addresses_consistent_with_particle.restype = ct.c_bool

# -------------------------------------------------------------------------------
# AssignAddressItem:

st_AssignAddressItem = stcom.st_AssignAddressItem
st_AssignAddressItem_p = stcom.st_AssignAddressItem_p
st_NullAssignAddressItem = stcom.st_NullAssignAddressItem

st_AssignAddressItem_print_out = testlib.st_AssignAddressItem_print_out_ext
st_AssignAddressItem_print_out.argtypes = [st_AssignAddressItem_p]
st_AssignAddressItem_print_out.restype = None
