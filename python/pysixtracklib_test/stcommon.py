from . import config as stconf
from pysixtracklib.particles import Particles as st_Particles
import pysixtracklib.stcommon as pyst

testlib = ct.CDLL( stconf.SHARED_LIB )

# C-API Types

st_Null           = pyst.st_Null
st_NullChar       = pyst.st_NullChar

st_uint64_p       = pyst.st_uint64_p
st_uchar_p        = pyst.st_uchar_p

st_double_p       = pyst.st_double_p
st_int64_p        = pyst.st_int64_p

st_Buffer         = pyst.st_Buffer
st_Buffer_p       = pyst.st_Buffer_p

# -----------------------------------------------------------------------------
# Particles realted functions and definitions

st_Particles      = pyst.st_Particles
st_Particles_p    = pyst.st_Particles_p
st_NullParticles  = pyst.st_NullParticles

st_Particles_realistic_init = testlib.st_Particles_realistic_init
st_Particles_realistic_init.argtypes = [ st_Particles_p ]
st_Particles_realistic_init.restype  = None

st_Particles_random_init   = testlib.st_Particles_random_init
st_Particles_random_init.argtypes = [ st_Particles_p ]
st_Particles_random_init.restype  = None

st_Particles_have_same_structure = testlib.st_Particles_have_same_structure
st_Particles_have_same_structure.argtypes = [ st_Particles_p, st_Particles_p ]
st_Particles_have_same_structure.restype  = ct.c_bool

st_Particles_map_to_same_memory = testlib.st_Particles_map_to_same_memory
st_Particles_map_to_same_memory.argtypes = [ st_Particles_p, st_Particles_p ]
st_Particles_map_to_same_memory.restype  = ct.c_bool

st_Particles_compare_real_values = testlib.st_Particles_compare_real_values
st_Particles_compare_real_values.restype  = ct.c_int32
st_Particles_compare_real_values.argtypes = [ st_Particles_p, st_Particles_p ]

st_Particles_compare_real_values_with_treshold = \
    testlib.st_Particles_compare_real_values_with_treshold
st_Particles_compare_real_values_with_treshold.restype  = ct.c_int32
st_Particles_compare_real_values_with_treshold.argtypes = [
        st_Particles_p, st_Particles_p ]

st_Particles_compare_integer_values = \
    testlib.st_Particles_compare_integer_values
st_Particles_compare_integer_values.restype  = ct.c_int32
st_Particles_compare_integer_values.argtypes = [
        st_Particles_p, st_Particles_p ]

st_Particles_compare_values = testlib.st_Particles_compare_values
st_Particles_compare_values.argtypes = [ st_Particles_p, st_Particles_p ]
st_Particles_compare_values.restypes = ct.c_int32

st_Particles_compare_values_with_treshold = \
    testlib.st_Particles_compare_values_with_treshold
st_Particles_compare_values_with_treshold.restype  = ct.c_int32
st_Particles_compare_values_with_treshold.argtypes = [
    st_Particles_p, st_Particles_p, ct.c_double ]

st_Particles_print_out_single = testlib.st_Particles_print_out_single
st_Particles_print_out_single.restype  = None
st_Particles_print_out_single.argtypes = [ st_Particles_p, ct.c_uint64 ]

st_Particles_print_out = testlib.st_Particles_print_out
st_Particles_print_out.restype  = None
st_Particles_print_out.argtypes = [ st_Particles_p ]

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_buffers_map_to_same_memory)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(Particles_buffers_compare_values)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(Particles_buffers_compare_values_with_treshold)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffers_get_max_difference)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_stdout)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_max_diff)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_out)(
     SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Particles_buffer_print_max_diff_out)(
     SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff,
     SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

