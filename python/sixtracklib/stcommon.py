import ctypes as ct
from . import config as stconf
from .particles import Particles as st_Particles
from .config import SIXTRACKLIB_MODULES
import cobjects
from cobjects import CBuffer

sixtracklib = ct.CDLL(stconf.SHARED_LIB)

# C-API Types

st_Null = ct.cast(0, ct.c_void_p)
st_NullChar = ct.cast(0, ct.c_char_p)

st_Context_p = ct.c_void_p
st_uint64_p = ct.POINTER(ct.c_uint64)
st_uchar_p = ct.POINTER(ct.c_ubyte)
st_const_uchar_p = ct.POINTER(ct.c_ubyte)

st_NullUChar = ct.cast(0, st_uchar_p)

st_double_p = ct.POINTER(ct.c_double)
st_int64_p = ct.POINTER(ct.c_int64)
st_buffer_size_t = ct.c_uint64
st_buffer_size_p = ct.POINTER(st_buffer_size_t)

st_arch_status_t = ct.c_int32
st_arch_size_t = ct.c_uint64
st_arch_size_t_p = ct.POINTER(st_arch_size_t)

st_ARCH_STATUS_SUCCESS = st_arch_status_t(0)
st_ARCH_STATUS_GENERAL_FAILURE = st_arch_status_t(-1)

# ------------------------------------------------------------------------------
# st_Buffer C-API functions


class st_Buffer(ct.Structure):
    _fields_ = [("data_addr", ct.c_uint64),
                ("data_size", ct.c_uint64),
                ("header_size", ct.c_uint64),
                ("data_capacity", ct.c_uint64),
                ("slot_length", ct.c_uint64),
                ("object_addr", ct.c_uint64),
                ("num_objects", ct.c_uint64),
                ("datastore_flags", ct.c_uint64),
                ("datastore_addr", ct.c_uint64)]


st_Buffer_p = ct.POINTER(st_Buffer)
st_NullBuffer = ct.cast(0, st_Buffer_p)
st_buffer_addr_t = ct.c_uint64

st_Buffer_preset = sixtracklib.st_Buffer_preset_ext
st_Buffer_preset.argtypes = [st_Buffer_p]
st_Buffer_preset.restype = st_Buffer_p

st_Buffer_new = sixtracklib.st_Buffer_new
st_Buffer_new.argtypes = [ct.c_uint64]
st_Buffer_new.restype = st_Buffer_p

st_Buffer_new_on_memory = sixtracklib.st_Buffer_new_on_memory
st_Buffer_new_on_memory.argtypes = [st_uchar_p, ct.c_uint64]
st_Buffer_new_on_memory.restype = st_Buffer_p

st_Buffer_new_on_data = sixtracklib.st_Buffer_new_on_data
st_Buffer_new_on_data.argtypes = [st_uchar_p, ct.c_uint64]
st_Buffer_new_on_data.restype = st_Buffer_p

st_Buffer_new_from_copy = sixtracklib.st_Buffer_new_from_copy
st_Buffer_new_from_copy.argtypes = [st_Buffer_p]
st_Buffer_new_from_copy.restype = st_Buffer_p

st_Buffer_new_from_file = sixtracklib.st_Buffer_new_from_file
st_Buffer_new_from_file.argtypes = [ct.c_char_p]
st_Buffer_new_from_file.restype = st_Buffer_p

st_Buffer_new_detailed = sixtracklib.st_Buffer_new_detailed
st_Buffer_new_detailed.argtypes = [
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t,
    ct.c_uint64]
st_Buffer_new_detailed.restype = st_Buffer_p

st_Buffer_init_from_data = sixtracklib.st_Buffer_init_from_data_ext
st_Buffer_init_from_data.argtypes = [st_Buffer_p, st_uchar_p, ct.c_uint64]
st_Buffer_init_from_data.restypes = ct.c_int32

st_Buffer_get_slot_size = sixtracklib.st_Buffer_get_slot_size_ext
st_Buffer_get_slot_size.argtypes = [st_Buffer_p, ]
st_Buffer_get_slot_size.restype = ct.c_uint64

st_Buffer_get_num_of_objects = sixtracklib.st_Buffer_get_num_of_objects_ext
st_Buffer_get_num_of_objects.argtypes = [st_Buffer_p, ]
st_Buffer_get_num_of_objects.restype = ct.c_uint64

st_Buffer_get_size = sixtracklib.st_Buffer_get_size_ext
st_Buffer_get_size.argtypes = [st_Buffer_p]
st_Buffer_get_size.restype = ct.c_uint64

st_Buffer_get_capacity = sixtracklib.st_Buffer_get_capacity_ext
st_Buffer_get_capacity.argtypes = [st_Buffer_p]
st_Buffer_get_capacity.restype = ct.c_uint64

st_Buffer_get_header_size = sixtracklib.st_Buffer_get_header_size_ext
st_Buffer_get_header_size.argtypes = [st_Buffer_p]
st_Buffer_get_header_size.restype = st_buffer_size_t

st_Buffer_get_data_begin_addr = sixtracklib.st_Buffer_get_data_begin_addr_ext
st_Buffer_get_data_begin_addr.argtypes = [st_Buffer_p]
st_Buffer_get_data_begin_addr.restype = st_buffer_addr_t

st_Buffer_get_data_end_addr = sixtracklib.st_Buffer_get_data_end_addr_ext
st_Buffer_get_data_end_addr.argtypes = [st_Buffer_p]
st_Buffer_get_data_end_addr.restype = st_buffer_addr_t

st_Buffer_get_objects_begin_addr = \
    sixtracklib.st_Buffer_get_objects_begin_addr_ext
st_Buffer_get_objects_begin_addr.argtypes = [st_Buffer_p]
st_Buffer_get_objects_begin_addr.restype = st_buffer_addr_t

st_Buffer_get_objects_end_addr = sixtracklib.st_Buffer_get_objects_end_addr_ext
st_Buffer_get_objects_end_addr.argtypes = [st_Buffer_p]
st_Buffer_get_objects_end_addr.restype = st_buffer_addr_t

st_Buffer_get_num_of_slots = sixtracklib.st_Buffer_get_num_of_slots_ext
st_Buffer_get_num_of_slots.argtypes = [st_Buffer_p]
st_Buffer_get_num_of_slots.restype = st_buffer_size_t

st_Buffer_get_max_num_of_slots = sixtracklib.st_Buffer_get_max_num_of_slots_ext
st_Buffer_get_max_num_of_slots.argtypes = [st_Buffer_p]
st_Buffer_get_max_num_of_slots.restype = st_buffer_size_t

st_Buffer_get_num_of_objects = sixtracklib.st_Buffer_get_num_of_objects_ext
st_Buffer_get_num_of_objects.argtypes = [st_Buffer_p]
st_Buffer_get_num_of_objects.restype = st_buffer_size_t

st_Buffer_get_max_num_of_objects = \
    sixtracklib.st_Buffer_get_max_num_of_objects_ext
st_Buffer_get_max_num_of_objects.argtypes = [st_Buffer_p]
st_Buffer_get_max_num_of_objects.restype = st_buffer_size_t

st_Buffer_get_num_of_dataptrs = sixtracklib.st_Buffer_get_num_of_dataptrs_ext
st_Buffer_get_num_of_dataptrs.argtypes = [st_Buffer_p]
st_Buffer_get_num_of_dataptrs.restype = st_buffer_size_t

st_Buffer_get_max_num_of_dataptrs = \
    sixtracklib.st_Buffer_get_max_num_of_dataptrs_ext
st_Buffer_get_max_num_of_dataptrs.argtypes = [st_Buffer_p]
st_Buffer_get_max_num_of_dataptrs.restype = st_buffer_size_t

st_Buffer_get_num_of_garbage_ranges = \
    sixtracklib.st_Buffer_get_num_of_garbage_ranges_ext
st_Buffer_get_num_of_garbage_ranges.argtypes = [st_Buffer_p]
st_Buffer_get_num_of_garbage_ranges.restype = st_buffer_size_t

st_Buffer_get_max_num_of_garbage_ranges = \
    sixtracklib.st_Buffer_get_max_num_of_garbage_ranges_ext
st_Buffer_get_max_num_of_garbage_ranges.argtypes = [st_Buffer_p]
st_Buffer_get_max_num_of_garbage_ranges.restype = st_buffer_size_t

st_Buffer_read_from_file = sixtracklib.st_Buffer_read_from_file
st_Buffer_read_from_file.argtypes = [st_Buffer_p]
st_Buffer_read_from_file.restype = ct.c_bool

st_Buffer_write_to_file = sixtracklib.st_Buffer_write_to_file
st_Buffer_write_to_file.argtypes = [st_Buffer_p, ct.c_char_p]
st_Buffer_write_to_file.restype = ct.c_bool

st_Buffer_write_to_file_normalized_addr = \
    sixtracklib.st_Buffer_write_to_file_normalized_addr
st_Buffer_write_to_file_normalized_addr.argtypes = [
    st_Buffer_p, ct.c_char_p, st_buffer_addr_t]
st_Buffer_write_to_file_normalized_addr.restype = ct.c_bool

st_Buffer_reset = sixtracklib.st_Buffer_reset_ext
st_Buffer_reset.argtypes = [st_Buffer_p]
st_Buffer_reset.restype = st_arch_status_t

st_Buffer_reset_detailed = sixtracklib.st_Buffer_reset_detailed_ext
st_Buffer_reset_detailed.argtypes = [
    st_Buffer_p,
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t]
st_Buffer_reset_detailed.restype = st_arch_status_t

st_Buffer_reserve = sixtracklib.st_Buffer_reserve_ext
st_Buffer_reserve.argtypes = [
    st_Buffer_p,
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t,
    st_buffer_size_t]
st_Buffer_reserve.restype = st_arch_status_t

st_Buffer_reserve_capacity = sixtracklib.st_Buffer_reserve_capacity_ext
st_Buffer_reserve_capacity.argtypes = [st_Buffer_p, st_buffer_size_t]
st_Buffer_reserve_capacity.restype = st_arch_status_t

st_Buffer_needs_remapping = sixtracklib.st_Buffer_needs_remapping_ext
st_Buffer_needs_remapping.argtypes = [st_Buffer_p]
st_Buffer_needs_remapping.restype = ct.c_bool

st_Buffer_remap = sixtracklib.st_Buffer_remap_ext
st_Buffer_remap.argtypes = [st_Buffer_p]
st_Buffer_remap.restype = st_arch_status_t

st_Buffer_refresh = sixtracklib.st_Buffer_refresh_ext
st_Buffer_refresh.argtypes = [st_Buffer_p]
st_Buffer_refresh.restype = st_arch_status_t

st_Buffer_clear = sixtracklib.st_Buffer_clear_ext
st_Buffer_clear.argtypes = [st_Buffer_p, ct.c_bool]
st_Buffer_clear.restype = st_arch_status_t

st_Buffer_free = sixtracklib.st_Buffer_free_ext
st_Buffer_free.argtypes = [st_Buffer_p]
st_Buffer_free.restype = None

st_Buffer_delete = sixtracklib.st_Buffer_delete
st_Buffer_delete.argtypes = [st_Buffer_p]
st_Buffer_delete.restype = None

# Helper Classes


def st_Buffer_new_mapped_on_cbuffer(cbuffer):
    data_ptr = ct.POINTER(ct.c_ubyte)
    ptr_data = ct.cast(cbuffer.base, data_ptr)
    size = ct.c_uint64(cbuffer.size)
    return st_Buffer_new_on_data(ptr_data, size)

# ------------------------------------------------------------------------------
# st_Particles C-API functions


class st_Particles(ct.Structure):
    _fields_ = [("num_particles", ct.c_int64), ("q0", st_double_p),
                ("mass0", st_double_p), ("beta0", st_double_p),
                ("gamma0", st_double_p), ("p0C", st_double_p),
                ("s", st_double_p), ("x", st_double_p), ("y", st_double_p),
                ("px", st_double_p), ("py", st_double_p), ("zeta", st_double_p),
                ("psigma", st_double_p), ("delta", st_double_p),
                ("rpp", st_double_p), ("rvv", st_double_p), ("chi", st_double_p),
                ("charge_ratio", st_double_p), ("particle_id", st_int64_p),
                ("at_element", st_int64_p), ("at_turn", st_int64_p),
                ("state", st_int64_p)]


class st_ParticlesAddr(ct.Structure):
    _fields_ = [("num_particles", ct.c_int64), ("q0", ct.c_uint64),
                ("mass0", ct.c_uint64), ("beta0", ct.c_uint64),
                ("gamma0", ct.c_uint64), ("p0C", ct.c_uint64),
                ("s", ct.c_uint64), ("x", ct.c_uint64), ("y", ct.c_uint64),
                ("px", ct.c_uint64), ("py", ct.c_uint64), ("zeta", ct.c_uint64),
                ("psigma", ct.c_uint64), ("delta", ct.c_uint64),
                ("rpp", ct.c_uint64), ("rvv", ct.c_uint64), ("chi", ct.c_uint64),
                ("charge_ratio", ct.c_uint64), ("particle_id", ct.c_uint64),
                ("at_element", ct.c_uint64), ("at_turn", ct.c_uint64),
                ("state", ct.c_uint64)]


st_Particles_p = ct.POINTER(st_Particles)
st_NullParticles = ct.cast(0, st_Particles_p)
st_particle_index_t = ct.c_int64
st_particle_real_t = ct.c_double
st_particle_num_elem_t = ct.c_int64

st_ParticlesAddr_p = ct.POINTER(st_ParticlesAddr)
st_NullParticlesAddr = ct.cast(0, st_ParticlesAddr_p)


def st_Particles_cbuffer_get_particles(cbuffer, obj_index):
    return ct.cast(cbuffer.get_object_address(obj_index), st_Particles_p)


st_Particles_preset = sixtracklib.st_Particles_preset_ext
st_Particles_preset.argtypes = [st_Particles_p]
st_Particles_preset.restype = st_Particles_p

st_Particles_get_num_of_particles = \
    sixtracklib.st_Particles_get_num_of_particles_ext
st_Particles_get_num_of_particles.argtypes = [st_Particles_p]
st_Particles_get_num_of_particles.restype = st_particle_num_elem_t

st_Particles_copy_single = sixtracklib.st_Particles_copy_single_ext
st_Particles_copy_single.restype = ct.c_bool
st_Particles_copy_single.argtypes = [
    st_Particles_p, st_particle_num_elem_t,
    st_Particles_p, st_particle_num_elem_t]

st_Particles_copy_range = sixtracklib.st_Particles_copy_range_ext
st_Particles_copy_range.restype = ct.c_bool
st_Particles_copy_range.argtypes = [
    st_Particles_p,
    st_Particles_p,
    st_particle_num_elem_t,
    st_particle_num_elem_t,
    st_particle_num_elem_t]

st_Particles_copy = sixtracklib.st_Particles_copy_ext
st_Particles_copy.argtypes = [st_Particles_p, st_Particles_p]
st_Particles_copy.restype = ct.c_bool

st_Particles_calculate_difference = \
    sixtracklib.st_Particles_calculate_difference_ext
st_Particles_calculate_difference.restype = None
st_Particles_calculate_difference.argtypes = [
    st_Particles_p, st_Particles_p, st_Particles_p]

st_Particles_buffer_get_total_num_of_particles = \
    sixtracklib.st_Particles_buffer_get_total_num_of_particles_ext
st_Particles_buffer_get_total_num_of_particles.restype = st_particle_num_elem_t
st_Particles_buffer_get_total_num_of_particles.argtypes = [st_Buffer_p]

st_Particles_buffer_get_num_of_particle_blocks = \
    sixtracklib.st_Particles_buffer_get_num_of_particle_blocks_ext
st_Particles_buffer_get_num_of_particle_blocks.restype = st_buffer_size_t
st_Particles_buffer_get_num_of_particle_blocks.argtypes = [st_Buffer_p]

st_Particles_buffer_get_particles = \
    sixtracklib.st_Particles_buffer_get_particles_ext
st_Particles_buffer_get_particles.restype = st_Particles_p
st_Particles_buffer_get_particles.argtypes = [st_Buffer_p, st_buffer_size_t]

st_Particles_buffers_have_same_structure = \
    sixtracklib.st_Particles_buffers_have_same_structure_ext
st_Particles_buffers_have_same_structure.restype = ct.c_bool
st_Particles_buffers_have_same_structure.argtypes = [
    st_Particles_p, st_Particles_p]

st_Particles_buffers_calculate_difference = \
    sixtracklib.st_Particles_buffers_calculate_difference_ext
st_Particles_buffers_calculate_difference.restype = None
st_Particles_buffers_calculate_difference.argtypes = [
    st_Buffer_p, st_Buffer_p, st_Buffer_p]

st_Particles_buffer_clear_particles = \
    sixtracklib.st_Particles_buffer_clear_particles_ext
st_Particles_buffer_clear_particles.restype = None
st_Particles_buffer_clear_particles.argtypes = [st_Buffer_p]

st_Particles_get_required_num_slots = \
    sixtracklib.st_Particles_get_required_num_slots_ext
st_Particles_get_required_num_slots.restype = st_buffer_size_t
st_Particles_get_required_num_slots.argtypes = [st_Buffer_p, st_buffer_size_t]

st_Particles_get_required_num_dataptrs = \
    sixtracklib.st_Particles_get_required_num_dataptrs_ext
st_Particles_get_required_num_dataptrs.restype = st_buffer_size_t
st_Particles_get_required_num_dataptrs.argtypes = [
    st_Buffer_p, st_buffer_size_t]

st_Particles_can_be_added = sixtracklib.st_Particles_can_be_added_ext
st_Particles_can_be_added.restype = ct.c_bool
st_Particles_can_be_added.argtypes = [
    st_Buffer_p, st_buffer_size_t, st_uint64_p, st_uint64_p, st_uint64_p]

st_Particles_new = sixtracklib.st_Particles_new_ext
st_Particles_new.argtypes = [st_Buffer_p, st_particle_num_elem_t]
st_Particles_new.restype = st_Particles_p

st_Particles_add = sixtracklib.st_Particles_add_ext
st_Particles_add.restype = st_Particles_p
st_Particles_add.argtypes = [
    st_Buffer_p,
    st_particle_num_elem_t,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_double_p,
    st_int64_p,
    st_int64_p,
    st_int64_p,
    st_int64_p]

st_Particles_add_copy = sixtracklib.st_Particles_add_copy_ext
st_Particles_add_copy.restype = st_Particles_p
st_Particles_add_copy.argtypes = [st_Buffer_p, st_Particles_p]

st_ParticlesAddr_preset = sixtracklib.st_ParticlesAddr_preset_ext
st_ParticlesAddr_preset.argtypes = [st_ParticlesAddr_p]
st_ParticlesAddr_preset.restype = st_ParticlesAddr_p

st_ParticlesAddr_assign_from_particles = \
    sixtracklib.st_ParticlesAddr_assign_from_particles_ext
st_ParticlesAddr_assign_from_particles.argtypes = [
    st_ParticlesAddr_p, st_Particles_p]
st_ParticlesAddr_assign_from_particles.restype = None

st_ParticlesAddr_assign_to_particles = \
    sixtracklib.st_ParticlesAddr_assign_to_particles_ext
st_ParticlesAddr_assign_to_particles.argtypes = [
    st_ParticlesAddr_p, st_Particles_p]
st_ParticlesAddr_assign_to_particles.restype = None

st_ParticlesAddr_remap_addresses = \
    sixtracklib.st_ParticlesAddr_remap_addresses_ext
st_ParticlesAddr_remap_addresses.argtypes = [st_ParticlesAddr_p, ct.c_int64]
st_ParticlesAddr_remap_addresses.restype = None

st_Buffer_is_particles_buffer = sixtracklib.st_Buffer_is_particles_buffer_ext
st_Buffer_is_particles_buffer.argtypes = [st_Buffer_p]
st_Buffer_is_particles_buffer.restype = ct.c_bool

# -----------------------------------------------------------------------------
# NS(ElemByelemConfig):

st_elem_by_elem_order_int_t = ct.c_int64
st_elem_by_elem_out_addr_t = ct.c_uint64
st_elem_by_elem_flag_t = ct.c_int64

st_ELEM_BY_ELEM_ORDER_INVALID = st_elem_by_elem_order_int_t(-1)
st_ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES = st_elem_by_elem_order_int_t(0)
st_ELEM_BY_ELEM_ORDER_DEFAULT = st_ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES


class st_ElemByElemConfig(ct.Structure):
    _fields_ = [("order", st_elem_by_elem_order_int_t),
                ("num_particles_to_store", st_particle_num_elem_t),
                ("num_elements_to_store", st_particle_num_elem_t),
                ("num_turns_to_store", st_particle_num_elem_t),
                ("min_particle_id", st_particle_index_t),
                ("min_element_id", st_particle_index_t),
                ("min_turn", st_particle_index_t),
                ("max_particle_id", st_particle_index_t),
                ("max_element_id", st_particle_index_t),
                ("max_turn", st_particle_index_t),
                ("is_rolling", st_elem_by_elem_flag_t),
                ("out_store_addr", st_elem_by_elem_out_addr_t)]


st_ElemByElemConfig_p = ct.POINTER(st_ElemByElemConfig)
st_NullElemByElemConfig = ct.cast(0, st_ElemByElemConfig_p)


st_ElemByElemConfig_create = sixtracklib.st_ElemByElemConfig_create
st_ElemByElemConfig_create.restype = st_ElemByElemConfig_p
st_ElemByElemConfig_create.argtypes = None

st_ElemByElemConfig_delete = sixtracklib.st_ElemByElemConfig_delete
st_ElemByElemConfig_delete.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_delete.restype = None

st_ElemByElemConfig_is_active = \
    sixtracklib.st_ElemByElemConfig_is_active_ext
st_ElemByElemConfig_is_active.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_is_active.restype = ct.c_bool

st_ElemByElemConfig_get_out_store_num_particles = \
    sixtracklib.st_ElemByElemConfig_get_out_store_num_particles_ext
st_ElemByElemConfig_get_out_store_num_particles.argtypes = [
    st_ElemByElemConfig_p]
st_ElemByElemConfig_get_out_store_num_particles.restype = \
    st_particle_num_elem_t

st_ElemByElemConfig_get_num_particles_to_store = \
    sixtracklib.st_ElemByElemConfig_get_num_particles_to_store_ext
st_ElemByElemConfig_get_num_particles_to_store.argtypes = [
    st_ElemByElemConfig_p]
st_ElemByElemConfig_get_num_particles_to_store.restype = st_particle_num_elem_t

st_ElemByElemConfig_get_num_turns_to_store = \
    sixtracklib.st_ElemByElemConfig_get_num_turns_to_store_ext
st_ElemByElemConfig_get_num_turns_to_store.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_num_turns_to_store.restype = st_particle_num_elem_t

st_ElemByElemConfig_get_num_elements_to_store = \
    sixtracklib.st_ElemByElemConfig_get_num_elements_to_store_ext
st_ElemByElemConfig_get_num_elements_to_store.argtypes = [
    st_ElemByElemConfig_p]
st_ElemByElemConfig_get_num_elements_to_store.restype = st_particle_num_elem_t

st_ElemByElemConfig_get_min_particle_id = \
    sixtracklib.st_ElemByElemConfig_get_min_particle_id_ext
st_ElemByElemConfig_get_min_particle_id.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_min_particle_id.restype = st_particle_index_t

st_ElemByElemConfig_get_max_particle_id = \
    sixtracklib.st_ElemByElemConfig_get_max_particle_id_ext
st_ElemByElemConfig_get_max_particle_id.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_max_particle_id.restype = st_particle_index_t

st_ElemByElemConfig_get_min_element_id = \
    sixtracklib.st_ElemByElemConfig_get_min_element_id_ext
st_ElemByElemConfig_get_min_element_id.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_min_element_id.restype = st_particle_index_t

st_ElemByElemConfig_get_max_element_id = \
    sixtracklib.st_ElemByElemConfig_get_max_element_id_ext
st_ElemByElemConfig_get_max_element_id.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_max_element_id.restype = st_particle_index_t

st_ElemByElemConfig_get_min_turn = \
    sixtracklib.st_ElemByElemConfig_get_min_turn_ext
st_ElemByElemConfig_get_min_turn.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_min_turn.restype = st_particle_index_t

st_ElemByElemConfig_get_max_turn = \
    sixtracklib.st_ElemByElemConfig_get_max_turn_ext
st_ElemByElemConfig_get_max_turn.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_max_turn.restype = st_particle_index_t

st_ElemByElemConfig_is_rolling = sixtracklib.st_ElemByElemConfig_is_rolling_ext
st_ElemByElemConfig_is_rolling.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_is_rolling.restype = ct.c_bool

st_ElemByElemConfig_get_order = sixtracklib.st_ElemByElemConfig_get_order_ext
st_ElemByElemConfig_get_order.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_get_order.restype = st_elem_by_elem_order_int_t

st_ElemByElemConfig_get_output_store_address = \
    sixtracklib.st_ElemByElemConfig_get_output_store_address_ext
st_ElemByElemConfig_get_output_store_address.argtypes = [
    st_ElemByElemConfig_p]
st_ElemByElemConfig_get_output_store_address.restype = \
    st_elem_by_elem_out_addr_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_ElemByElemConfig_get_particles_store_index_details = \
    sixtracklib.st_ElemByElemConfig_get_particles_store_index_details_ext
st_ElemByElemConfig_get_particles_store_index_details.argtypes = [
    st_ElemByElemConfig_p, st_particle_index_t, st_particle_index_t,
    st_particle_index_t]
st_ElemByElemConfig_get_particles_store_index_details.restype = \
    st_particle_num_elem_t

st_ElemByElemConfig_get_particles_store_index = \
    sixtracklib.st_ElemByElemConfig_get_particles_store_index_ext
st_ElemByElemConfig_get_particles_store_index.argtypes = [
    st_ElemByElemConfig_p, st_Particles_p, st_particle_num_elem_t]
st_ElemByElemConfig_get_particles_store_index.restype = st_particle_num_elem_t

st_ElemByElemConfig_get_particle_id_from_store_index = \
    sixtracklib.st_ElemByElemConfig_get_particle_id_from_store_index_ext
st_ElemByElemConfig_get_particle_id_from_store_index.argtypes = [
    st_ElemByElemConfig_p, st_particle_num_elem_t]
st_ElemByElemConfig_get_particle_id_from_store_index.restype = \
    st_particle_num_elem_t

st_ElemByElemConfig_get_at_element_id_from_store_index = \
    sixtracklib.st_ElemByElemConfig_get_at_element_id_from_store_index_ext
st_ElemByElemConfig_get_at_element_id_from_store_index.argtypes = [
    st_ElemByElemConfig_p, st_particle_num_elem_t]
st_ElemByElemConfig_get_at_element_id_from_store_index.restype = \
    st_particle_num_elem_t

st_ElemByElemConfig_get_at_turn_from_store_index = \
    sixtracklib.st_ElemByElemConfig_get_at_turn_from_store_index_ext
st_ElemByElemConfig_get_at_turn_from_store_index.argtypes = [
    st_ElemByElemConfig_p, st_particle_num_elem_t]
st_ElemByElemConfig_get_at_turn_from_store_index.restype = st_particle_index_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_ElemByElemConfig_init = sixtracklib.st_ElemByElemConfig_init
st_ElemByElemConfig_init.argtypes = [
    st_ElemByElemConfig_p,
    st_Particles_p,
    st_Buffer_p,
    st_particle_index_t,
    st_particle_index_t]
st_ElemByElemConfig_init.restype = st_arch_status_t

st_ElemByElemConfig_init_on_particle_sets = \
    sixtracklib.st_ElemByElemConfig_init_on_particle_sets
st_ElemByElemConfig_init_on_particle_sets.argtypes = [
    st_ElemByElemConfig_p,
    st_Buffer_p,
    st_buffer_size_t,
    st_buffer_size_p,
    st_Buffer_p,
    st_particle_index_t,
    st_particle_index_t]
st_ElemByElemConfig_init_on_particle_sets.restype = st_arch_status_t

st_ElemByElemConfig_init_detailed = \
    sixtracklib.st_ElemByElemConfig_init_detailed_ext
st_ElemByElemConfig_init_detailed.argtypes = [
    st_ElemByElemConfig_p,
    st_elem_by_elem_order_int_t,
    st_particle_index_t,
    st_particle_index_t,
    st_particle_index_t,
    st_particle_index_t,
    st_particle_index_t,
    st_particle_index_t,
    ct.c_bool]
st_ElemByElemConfig_init_detailed.restype = st_arch_status_t

st_ElemByElemConfig_assign_output_buffer = \
    sixtracklib.st_ElemByElemConfig_assign_output_buffer
st_ElemByElemConfig_assign_output_buffer.restype = st_arch_status_t
st_ElemByElemConfig_assign_output_buffer.argtypes = [
    st_ElemByElemConfig_p, st_Buffer_p, st_buffer_size_t]


def st_ElemByElemConfig_assign_output_cbuffer(
        elem_by_elem_config,
        output_buffer,
        out_buffer_offset_index):
    ptr_output_buffer = st_Buffer_new_mapped_on_cbuffer(output_buffer)

    out_buffer_offset_index_arg = st_buffer_size_t(out_buffer_offset_index)

    ret = st_ElemByElemConfig_assign_output_buffer(
        elem_by_elem_config, ptr_output_buffer, out_buffer_offset_index_arg)

    st_Buffer_delete(ptr_output_buffer)
    ptr_output_buffer = st_NullBuffer
    return ret

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


st_ElemByElemConfig_preset = sixtracklib.st_ElemByElemConfig_preset_ext
st_ElemByElemConfig_preset.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_preset.restype = st_ElemByElemConfig_p

st_ElemByElemConfig_clear = sixtracklib.st_ElemByElemConfig_clear_ext
st_ElemByElemConfig_clear.argtypes = [st_ElemByElemConfig_p]
st_ElemByElemConfig_clear.restype = None

st_ElemByElemConfig_set_order = sixtracklib.st_ElemByElemConfig_set_order_ext
st_ElemByElemConfig_set_order.argtypes = [
    st_ElemByElemConfig_p, st_elem_by_elem_order_int_t]
st_ElemByElemConfig_set_order.restype = None

st_ElemByElemConfig_set_is_rolling = \
    sixtracklib.st_ElemByElemConfig_set_is_rolling_ext
st_ElemByElemConfig_set_is_rolling.argtypes = [
    st_ElemByElemConfig_p, ct.c_bool]
st_ElemByElemConfig_set_is_rolling.restype = None

st_ElemByElemConfig_set_output_store_address = \
    sixtracklib.st_ElemByElemConfig_set_output_store_address_ext
st_ElemByElemConfig_set_output_store_address.argtypes = [
    st_ElemByElemConfig_p, st_elem_by_elem_out_addr_t]
st_ElemByElemConfig_set_output_store_address.restype = None

# -----------------------------------------------------------------------------
# BeamMonitor objects

st_BeamMonitor_insert_end_of_turn_monitors = \
    sixtracklib.st_BeamMonitor_insert_end_of_turn_monitors_at_pos
st_BeamMonitor_insert_end_of_turn_monitors.restype = ct.c_int32
st_BeamMonitor_insert_end_of_turn_monitors.argtypes = [
    st_Buffer_p, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_uint64, ]

st_BeamMonitor_assign_output_buffer = \
    sixtracklib.st_BeamMonitor_assign_output_buffer
st_BeamMonitor_assign_output_buffer.restype = ct.c_int32
st_BeamMonitor_assign_output_buffer.argtypes = [
    st_Buffer_p, st_Buffer_p, ct.c_int64, ct.c_uint64]


def st_BeamMonitor_assign_output_cbuffer(
        beam_elements_buffer,
        output_buffer,
        min_turn_id,
        until_turn_elem_by_elem):
    ptr_belem_buffer = st_Buffer_new_mapped_on_cbuffer(beam_elements_buffer)
    ptr_output_buffer = st_Buffer_new_mapped_on_cbuffer(output_buffer)

    min_turn_id = ct.c_int64(min_turn_id)
    until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)

    ret = st_BeamMonitor_assign_output_buffer(
        ptr_belem_buffer, ptr_output_buffer,
        min_turn_id, until_turn_elem_by_elem)

    st_Buffer_delete(ptr_belem_buffer)
    st_Buffer_delete(ptr_output_buffer)

    return ret


st_BeamMonitor_assign_output_buffer_from_offset = \
    sixtracklib.st_BeamMonitor_assign_output_buffer_from_offset
st_BeamMonitor_assign_output_buffer_from_offset.restype = ct.c_int32
st_BeamMonitor_assign_output_buffer_from_offset.argtypes = [
    st_Buffer_p, st_Buffer_p, ct.c_int64, ct.c_uint64]

# -----------------------------------------------------------------------------
# OutputBuffer bindings

st_out_buffer_flags_t = ct.c_int32

st_OutputBuffer_requires_output_buffer = \
    sixtracklib.st_OutputBuffer_requires_output_buffer_ext
st_OutputBuffer_requires_output_buffer.restype = ct.c_bool
st_OutputBuffer_requires_output_buffer.argtypes = [ct.c_int32]

st_OutputBuffer_requires_beam_monitor_output = \
    sixtracklib.st_OutputBuffer_requires_beam_monitor_output_ext
st_OutputBuffer_requires_beam_monitor_output.restype = ct.c_bool
st_OutputBuffer_requires_beam_monitor_output.argtypes = [ct.c_int32]

st_OutputBuffer_requires_elem_by_elem_output = \
    sixtracklib.st_OutputBuffer_requires_elem_by_elem_output_ext
st_OutputBuffer_requires_elem_by_elem_output.restype = ct.c_bool
st_OutputBuffer_requires_elem_by_elem_output.argtypes = [ct.c_int32]

st_OutputBuffer_required_for_tracking = \
    sixtracklib.st_OutputBuffer_required_for_tracking
st_OutputBuffer_required_for_tracking.restype = ct.c_int32
st_OutputBuffer_required_for_tracking.argtypes = [
    st_Particles_p,
    st_Buffer_p,
    ct.c_uint64]

st_OutputBuffer_required_for_tracking_of_particle_sets = \
    sixtracklib.st_OutputBuffer_required_for_tracking_of_particle_sets
st_OutputBuffer_required_for_tracking_of_particle_sets.restype = ct.c_int32
st_OutputBuffer_required_for_tracking_of_particle_sets.argtypes = [
    st_Buffer_p,
    ct.c_uint64,
    st_uint64_p,
    st_Buffer_p,
    ct.c_uint64]

st_OutputBuffer_prepare = sixtracklib.st_OutputBuffer_prepare
st_OutputBuffer_prepare.restype = ct.c_int32
st_OutputBuffer_prepare.argtypes = [
    st_Buffer_p,
    st_Buffer_p,
    st_Particles_p,
    ct.c_uint64,
    st_uint64_p,
    st_uint64_p,
    st_int64_p]

st_OutputBuffer_prepare_for_particle_sets = \
    sixtracklib.st_OutputBuffer_prepare_for_particle_sets
st_OutputBuffer_prepare_for_particle_sets.restype = ct.c_int32
st_OutputBuffer_prepare_for_particle_sets.argtypes = [
    st_Buffer_p,
    st_Buffer_p,
    st_Buffer_p,
    ct.c_uint64,
    st_uint64_p,
    ct.c_uint64,
    st_uint64_p,
    st_uint64_p,
    st_uint64_p]


st_OutputBuffer_calculate_output_buffer_params = \
    sixtracklib.st_OutputBuffer_calculate_output_buffer_params
st_OutputBuffer_calculate_output_buffer_params.restype = ct.c_int32
st_OutputBuffer_calculate_output_buffer_params.argtypes = [
    st_Buffer_p,
    st_Particles_p,
    ct.c_uint64,
    st_uint64_p,
    st_uint64_p,
    st_uint64_p,
    st_uint64_p,
    ct.c_uint64]


st_OutputBuffer_calculate_output_buffer_params_for_particles_sets = \
    sixtracklib.st_OutputBuffer_calculate_output_buffer_params_for_particles_sets
st_OutputBuffer_calculate_output_buffer_params_for_particles_sets.argtypes = [
    st_Buffer_p, st_Buffer_p, ct.c_uint64, st_uint64_p, ct.c_uint64,
    st_uint64_p, st_uint64_p, st_uint64_p, st_uint64_p, ct.c_uint64]
st_OutputBuffer_calculate_output_buffer_params.restype = ct.c_int32


def st_OutputBuffer_create_output_cbuffer(
        beam_elements_buffer,
        particles_buffer,
        num_particle_sets=1,
        indices=None,
        until_turn_elem_by_elem=0):

    ptr_particles_buffer = st_Buffer_new_mapped_on_cbuffer(particles_buffer)
    particles = st_Particles_buffer_get_particles(ptr_particles_buffer, 0)

    ptr_beam_elements_buffer = \
        st_Buffer_new_mapped_on_cbuffer(beam_elements_buffer)

    num_objects = ct.c_uint64(0)
    num_slots = ct.c_uint64(0)
    num_dataptrs = ct.c_uint64(0)
    num_garbage = ct.c_uint64(0)
    slot_size = st_Buffer_get_slot_size(ptr_particles_buffer)
    until_turn_elem_by_elem = ct.c_uint64(until_turn_elem_by_elem)

    ret = st_OutputBuffer_calculate_output_buffer_params(
        ptr_beam_elements_buffer, particles,
        until_turn_elem_by_elem, ct.byref(num_objects),
        ct.byref(num_slots), ct.byref(num_dataptrs),
        ct.byref(num_garbage), slot_size)

    output_buffer = None
    elem_by_elem_output_offset = -1
    beam_monitor_output_offset = -1
    min_turn_id = -1

    if ret == 0 and num_objects.value > 0 and num_slots.value > 0 and \
            num_dataptrs.value > 0 and num_garbage.value >= 0:
        if output_buffer is None:
            output_buffer = CBuffer(max_slots=num_slots.value,
                                    max_objects=num_objects.value,
                                    max_pointers=num_dataptrs.value,
                                    max_garbage=num_garbage.value)

        ptr_out_buffer_view = st_Buffer_new_mapped_on_cbuffer(output_buffer)

        elem_by_elem_output_offset = ct.c_uint64(0)
        beam_monitor_output_offset = ct.c_uint64(0)
        min_turn_id = ct.c_int64(0)

        if ptr_out_buffer_view != st_NullBuffer:
            ret = st_OutputBuffer_prepare(
                ptr_beam_elements_buffer,
                ptr_out_buffer_view,
                particles,
                until_turn_elem_by_elem,
                ct.byref(elem_by_elem_output_offset),
                ct.byref(beam_monitor_output_offset),
                ct.byref(min_turn_id))

        st_Buffer_delete(ptr_out_buffer_view)

    st_Buffer_delete(ptr_beam_elements_buffer)
    st_Buffer_delete(ptr_particles_buffer)

    return (output_buffer, elem_by_elem_output_offset.value,
            beam_monitor_output_offset.value, min_turn_id.value)

# -----------------------------------------------------------------------------
# TrackJob objects


st_TrackJob_p = ct.c_void_p
st_NullTrackJob = ct.cast(0, st_TrackJob_p)

st_TrackJob_create = sixtracklib.st_TrackJobCpu_create
st_TrackJob_create.argtypes = [ct.c_char_p, ct.c_char_p]
st_TrackJob_create.restype = st_TrackJob_p


st_TrackJob_new = sixtracklib.st_TrackJob_new
st_TrackJob_new.argtypes = [ct.c_char_p, st_Buffer_p, st_Buffer_p, ct.c_char_p]
st_TrackJob_new.restype = st_TrackJob_p

st_TrackJob_reset = sixtracklib.st_TrackJob_reset_with_output
st_TrackJob_reset.restype = ct.c_bool
st_TrackJob_reset.argtypes = [st_TrackJob_p, st_Buffer, st_Buffer, st_Buffer]

st_TrackJob_reset_with_output = sixtracklib.st_TrackJob_reset_with_output
st_TrackJob_reset_with_output.restype = ct.c_bool
st_TrackJob_reset_with_output.argtypes = [
    st_TrackJob_p, st_Buffer, st_Buffer, st_Buffer, ct.c_uint64]

st_TrackJob_reset_detailed = sixtracklib.st_TrackJob_reset_detailed
st_TrackJob_reset_detailed.restype = ct.c_bool
st_TrackJob_reset_detailed.argtypes = [
    st_TrackJob_p, st_Buffer, ct.c_uint64, st_uint64_p, st_Buffer,
    st_Buffer, ct.c_uint64]


st_TrackJob_new_with_output = sixtracklib.st_TrackJob_new_with_output
st_TrackJob_new_with_output.argtypes = [
    ct.c_char_p, st_Buffer_p, st_Buffer_p, st_Buffer_p,
    ct.c_uint64, ct.c_char_p]
st_TrackJob_new_with_output.restype = st_TrackJob_p


st_TrackJob_delete = sixtracklib.st_TrackJob_delete
st_TrackJob_delete.argtypes = [st_TrackJob_p]
st_TrackJob_delete.restype = None


st_TrackJob_track_until = sixtracklib.st_TrackJob_track_until
st_TrackJob_track_until.argtypes = [st_TrackJob_p, ct.c_uint64]
st_TrackJob_track_until.restype = ct.c_int32


st_TrackJob_track_elem_by_elem = sixtracklib.st_TrackJob_track_elem_by_elem
st_TrackJob_track_elem_by_elem.argtypes = [st_TrackJob_p, ct.c_uint64]
st_TrackJob_track_elem_by_elem.restype = ct.c_int32


st_TrackJob_track_line = sixtracklib.st_TrackJob_track_line
st_TrackJob_track_line.restype = ct.c_int32
st_TrackJob_track_line.argtypes = [
    st_TrackJob_p,
    ct.c_uint64,
    ct.c_uint64,
    ct.c_bool]


st_TrackJob_collect = sixtracklib.st_TrackJob_collect
st_TrackJob_collect.argtypes = [st_TrackJob_p]
st_TrackJob_collect.restype = None

st_TrackJob_collect_detailed = sixtracklib.st_TrackJob_collect_detailed
st_TrackJob_collect_detailed.argtypes = [st_TrackJob_p, ct.c_uint16]
st_TrackJob_collect_detailed.restype = None

st_TrackJob_collect_particles = sixtracklib.st_TrackJob_collect_particles
st_TrackJob_collect_particles.argtypes = [st_TrackJob_p]
st_TrackJob_collect_particles.restype = None

st_TrackJob_collect_beam_elements = sixtracklib.st_TrackJob_collect_beam_elements
st_TrackJob_collect_beam_elements.argtypes = [st_TrackJob_p]
st_TrackJob_collect_beam_elements.restype = None

st_TrackJob_collect_output = sixtracklib.st_TrackJob_collect_output
st_TrackJob_collect_output.argtypes = [st_TrackJob_p]
st_TrackJob_collect_output.restype = None

st_TrackJob_requires_collecting = sixtracklib.st_TrackJob_requires_collecting
st_TrackJob_requires_collecting.argtypes = [st_TrackJob_p]
st_TrackJob_requires_collecting.restype = ct.c_bool

st_TrackJob_push = sixtracklib.st_TrackJob_push
st_TrackJob_push.argtypes = [st_TrackJob_p, ct.c_uint16]
st_TrackJob_push.restype = None

st_TrackJob_push_particles = sixtracklib.st_TrackJob_push_particles
st_TrackJob_push_particles.argtypes = [st_TrackJob_p]
st_TrackJob_push_particles.restype = None

st_TrackJob_push_beam_elements = sixtracklib.st_TrackJob_push_beam_elements
st_TrackJob_push_beam_elements.argtypes = [st_TrackJob_p]
st_TrackJob_push_beam_elements.restype = None

st_TrackJob_push_output = sixtracklib.st_TrackJob_push_output
st_TrackJob_push_output.argtypes = [st_TrackJob_p]
st_TrackJob_push_output.restype = None

st_TrackJob_can_fetch_particle_addresses = \
    sixtracklib.st_TrackJob_can_fetch_particle_addresses
st_TrackJob_can_fetch_particle_addresses.argtypes = [st_TrackJob_p]
st_TrackJob_can_fetch_particle_addresses.restype = ct.c_bool

st_TrackJob_has_particle_addresses = \
    sixtracklib.st_TrackJob_has_particle_addresses
st_TrackJob_has_particle_addresses.argtypes = [st_TrackJob_p]
st_TrackJob_has_particle_addresses.restype = ct.c_bool

st_TrackJob_get_type_id = sixtracklib.st_TrackJob_get_type_id
st_TrackJob_get_type_id.argtypes = [st_TrackJob_p]
st_TrackJob_get_type_id.restype = ct.c_int64


st_TrackJob_get_type_str = sixtracklib.st_TrackJob_get_type_str
st_TrackJob_get_type_str.argtypes = [st_TrackJob_p]
st_TrackJob_get_type_str.restype = ct.c_char_p


st_TrackJob_has_output_buffer = sixtracklib.st_TrackJob_has_output_buffer
st_TrackJob_has_output_buffer.argtypes = [st_TrackJob_p]
st_TrackJob_has_output_buffer.restype = ct.c_bool


st_TrackJob_owns_output_buffer = sixtracklib.st_TrackJob_owns_output_buffer
st_TrackJob_owns_output_buffer.argtypes = [st_TrackJob_p]
st_TrackJob_owns_output_buffer.restype = ct.c_bool


st_TrackJob_get_output_buffer = sixtracklib.st_TrackJob_get_output_buffer
st_TrackJob_get_output_buffer.argtypes = [st_TrackJob_p]
st_TrackJob_get_output_buffer.restype = st_Buffer_p


st_TrackJob_has_elem_by_elem_output = \
    sixtracklib.st_TrackJob_has_elem_by_elem_output
st_TrackJob_has_elem_by_elem_output.argtypes = [st_TrackJob_p]
st_TrackJob_has_elem_by_elem_output.restype = ct.c_bool


st_TrackJob_get_elem_by_elem_output_buffer_offset = \
    sixtracklib.st_TrackJob_get_elem_by_elem_output_buffer_offset
st_TrackJob_get_elem_by_elem_output_buffer_offset.argtypes = [st_TrackJob_p]
st_TrackJob_get_elem_by_elem_output_buffer_offset.restype = ct.c_uint64


st_TrackJob_has_beam_monitor_output = \
    sixtracklib.st_TrackJob_has_beam_monitor_output
st_TrackJob_has_beam_monitor_output.argtypes = [st_TrackJob_p]
st_TrackJob_has_beam_monitor_output.restype = ct.c_bool


st_TrackJob_get_num_beam_monitors = \
    sixtracklib.st_TrackJob_get_num_beam_monitors
st_TrackJob_get_num_beam_monitors.argtypes = [st_TrackJob_p]
st_TrackJob_get_num_beam_monitors.restype = ct.c_uint64


st_TrackJob_get_beam_monitor_output_buffer_offset = \
    sixtracklib.st_TrackJob_get_beam_monitor_output_buffer_offset
st_TrackJob_get_beam_monitor_output_buffer_offset.argtypes = [st_TrackJob_p]
st_TrackJob_get_beam_monitor_output_buffer_offset.restype = ct.c_uint64

# ==============================================================================
# sixtracklib/control, sixtracklib/track API:

st_arch_id_t = ct.c_uint64
st_node_platform_id_t = ct.c_int64
st_node_device_id_t = ct.c_int64
st_node_index_t = ct.c_uint32
st_node_index_p = ct.POINTER(st_node_index_t)

# ------------------------------------------------------------------------------
# NS(NodeId):
st_NodeId_p = ct.c_void_p
st_NullNodeId = ct.cast(0, st_NodeId_p)

st_NodeId_create = sixtracklib.st_NodeId_create
st_NodeId_create.argtypes = None
st_NodeId_create.restype = st_NodeId_p

st_NodeId_new = sixtracklib.st_NodeId_new
st_NodeId_new.argtypes = [st_node_platform_id_t, st_node_device_id_t]
st_NodeId_new.restype = st_NodeId_p

st_NodeId_new_from_string = sixtracklib.st_NodeId_new_from_string
st_NodeId_new_from_string.argtypes = [ct.c_char_p]
st_NodeId_new_from_string.restype = st_NodeId_p

st_NodeId_new_detailed = sixtracklib.st_NodeId_new_detailed
st_NodeId_new_detailed.argtypes = [
    st_node_platform_id_t, st_node_device_id_t, st_node_index_t]
st_NodeId_new_detailed.restype = st_NodeId_p

st_NodeId_delete = sixtracklib.st_NodeId_delete
st_NodeId_delete.argtypes = [st_NodeId_p]
st_NodeId_delete.restype = None

st_NodeId_preset = sixtracklib.st_NodeId_preset
st_NodeId_preset.argtypes = [st_NodeId_p]
st_NodeId_preset.restype = st_NodeId_p

st_NodeId_is_valid = sixtracklib.st_NodeId_is_valid
st_NodeId_is_valid.argtypes = [st_NodeId_p]
st_NodeId_is_valid.restype = ct.c_bool


st_NodeId_get_platform_id = sixtracklib.st_NodeId_get_platform_id
st_NodeId_get_platform_id.argtypes = [st_NodeId_p]
st_NodeId_get_platform_id.restype = ct.c_int64

st_NodeId_get_device_id = sixtracklib.st_NodeId_get_device_id
st_NodeId_get_device_id.argtypes = [st_NodeId_p]
st_NodeId_get_device_id.restype = ct.c_int64

st_NodeId_has_node_index = sixtracklib.st_NodeId_has_node_index
st_NodeId_has_node_index.argtypes = [st_NodeId_p]
st_NodeId_has_node_index.restype = ct.c_bool

st_NodeId_get_node_index = sixtracklib.st_NodeId_get_node_index
st_NodeId_get_node_index.argtypes = [st_NodeId_p]
st_NodeId_get_node_index.restype = ct.c_uint32

st_NodeId_clear = sixtracklib.st_NodeId_clear
st_NodeId_clear.argtypes = [st_NodeId_p]
st_NodeId_clear.restype = None

st_NodeId_reset = sixtracklib.st_NodeId_reset
st_NodeId_reset.argtypes = [
    st_NodeId_p, st_node_platform_id_t, st_node_device_id_t, st_node_index_t]

_st_NodeId_to_string = sixtracklib.st_NodeId_to_string
_st_NodeId_to_string.argtypes = [st_NodeId_p, ct.c_char_p, st_buffer_size_t]
_st_NodeId_to_string.restype = st_arch_status_t

st_NodeId_compare = sixtracklib.st_NodeId_compare
st_NodeId_compare.argtypes = [st_NodeId_p]
st_NodeId_compare.restype = ct.c_int

st_NodeId_are_equal = sixtracklib.st_NodeId_are_equal
st_NodeId_are_equal.argtypes = [st_NodeId_p, st_NodeId_p]
st_NodeId_are_equal.restype = ct.c_bool

st_NodeId_print_out = sixtracklib.st_NodeId_print_out
st_NodeId_print_out.argtypes = [st_NodeId_p]
st_NodeId_print_out.restype = None

# -----------------------------------------------------------------------------
# NS(NodeInfo):

st_NodeInfoBase_p = ct.c_void_p
st_NullNodeInfoBase = ct.cast(0, st_NodeInfoBase_p)

st_NodeInfo_delete = sixtracklib.st_NodeInfo_delete
st_NodeInfo_delete.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_delete.restype = None

st_NodeInfo_get_ptr_const_node_id = \
    sixtracklib.st_NodeInfo_get_ptr_const_node_id
st_NodeInfo_get_ptr_const_node_id.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_ptr_const_node_id.restype = st_NodeId_p

st_NodeInfo_get_platform_id = sixtracklib.st_NodeInfo_get_platform_id
st_NodeInfo_get_platform_id.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_platform_id.restype = st_node_platform_id_t

st_NodeInfo_get_device_id = sixtracklib.st_NodeInfo_get_device_id
st_NodeInfo_get_device_id.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_device_id.restype = st_node_device_id_t

st_NodeInfo_has_node_index = sixtracklib.st_NodeInfo_has_node_index
st_NodeInfo_has_node_index.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_has_node_index.restype = ct.c_bool

st_NodeInfo_get_node_index = sixtracklib.st_NodeInfo_get_node_index
st_NodeInfo_get_node_index.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_node_index.restype = st_node_index_t

st_NodeInfo_is_default_node = sixtracklib.st_NodeInfo_is_default_node
st_NodeInfo_is_default_node.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_is_default_node.restype = ct.c_bool

st_NodeInfo_is_selected_node = sixtracklib.st_NodeInfo_is_selected_node
st_NodeInfo_is_selected_node.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_is_selected_node.restype = ct.c_bool

st_NodeInfo_get_arch_id = sixtracklib.st_NodeInfo_get_arch_id
st_NodeInfo_get_arch_id.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_arch_id.restype = st_arch_id_t

st_NodeInfo_has_arch_string = sixtracklib.st_NodeInfo_has_arch_string
st_NodeInfo_has_arch_string.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_has_arch_string.restype = ct.c_bool

st_NodeInfo_get_arch_string = sixtracklib.st_NodeInfo_get_arch_string
st_NodeInfo_get_arch_string.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_arch_string.restype = ct.c_char_p

st_NodeInfo_has_platform_name = sixtracklib.st_NodeInfo_has_platform_name
st_NodeInfo_has_platform_name.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_has_platform_name.restype = ct.c_bool

st_NodeInfo_get_platform_name = sixtracklib.st_NodeInfo_get_platform_name
st_NodeInfo_get_platform_name.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_platform_name.restype = ct.c_char_p

st_NodeInfo_has_device_name = sixtracklib.st_NodeInfo_has_device_name
st_NodeInfo_has_device_name.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_has_device_name.restype = ct.c_bool

st_NodeInfo_get_device_name = sixtracklib.st_NodeInfo_get_device_name
st_NodeInfo_get_device_name.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_device_name.restype = ct.c_char_p

st_NodeInfo_has_description = sixtracklib.st_NodeInfo_has_description
st_NodeInfo_has_description.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_has_description.restype = ct.c_bool

st_NodeInfo_get_description = sixtracklib.st_NodeInfo_get_description
st_NodeInfo_get_description.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_description.restype = ct.c_char_p

st_NodeInfo_print_out = sixtracklib.st_NodeInfo_print_out
st_NodeInfo_print_out.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_print_out.restype = None

st_NodeInfo_get_required_output_str_length = \
    sixtracklib.st_NodeInfo_get_required_output_str_length
st_NodeInfo_get_required_output_str_length.argtypes = [st_NodeInfoBase_p]
st_NodeInfo_get_required_output_str_length.restype = st_arch_size_t

_st_NodeInfo_convert_to_string = sixtracklib.st_NodeInfo_convert_to_string
_st_NodeInfo_convert_to_string.argtypes = [
    st_NodeInfoBase_p, st_arch_size_t, ct.c_char_p]
_st_NodeInfo_convert_to_string.restype = st_arch_status_t

# -----------------------------------------------------------------------------
# NS(KernelConfigBase)

st_KernelConfigBase_p = ct.c_void_p
st_NullKernelConfigBase = ct.cast(0, st_KernelConfigBase_p)
st_kernel_id_t = ct.c_uint32

st_ctrl_size_t = ct.c_uint64
st_ctrl_size_t_p = ct.POINTER(st_ctrl_size_t)
st_ctrl_status_t = ct.c_int32

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_delete = sixtracklib.st_KernelConfig_delete
st_KernelConfig_delete.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_delete.restype = None

st_KernelConfig_get_arch_id = sixtracklib.st_KernelConfig_get_arch_id
st_KernelConfig_get_arch_id.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_arch_id.restype = st_arch_id_t

st_KernelConfig_has_arch_string = sixtracklib.st_KernelConfig_has_arch_string
st_KernelConfig_has_arch_string.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_has_arch_string.restype = ct.c_bool

_st_KernelConfig_get_ptr_arch_string = \
    sixtracklib.st_KernelConfig_get_ptr_arch_string
_st_KernelConfig_get_ptr_arch_string.argtypes = [st_KernelConfigBase_p]
_st_KernelConfig_get_ptr_arch_string.restype = ct.c_char_p

st_KernelConfig_has_kernel_id = sixtracklib.st_KernelConfig_has_kernel_id
st_KernelConfig_has_kernel_id.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_has_kernel_id.restype = ct.c_bool

st_KernelConfig_get_kernel_id = sixtracklib.st_KernelConfig_get_kernel_id
st_KernelConfig_get_kernel_id.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_kernel_id.restype = st_kernel_id_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_has_name = sixtracklib.st_KernelConfig_has_name
st_KernelConfig_has_name.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_has_name.restype = ct.c_bool

_st_KernelConfig_get_ptr_name_string = \
    sixtracklib.st_KernelConfig_get_ptr_name_string
_st_KernelConfig_get_ptr_name_string.argtypes = [st_KernelConfigBase_p]
_st_KernelConfig_get_ptr_name_string.restype = ct.c_char_p

st_KernelConfig_get_num_arguments = \
    sixtracklib.st_KernelConfig_get_num_arguments
st_KernelConfig_get_num_arguments.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_num_arguments.restype = st_ctrl_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_get_work_items_dim = \
    sixtracklib.st_KernelConfig_get_work_items_dim
st_KernelConfig_get_work_items_dim.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_work_items_dim.restype = st_ctrl_size_t

st_KernelConfig_get_num_work_items_by_dim = \
    sixtracklib.st_KernelConfig_get_num_work_items_by_dim
st_KernelConfig_get_num_work_items_by_dim.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t]
st_KernelConfig_get_num_work_items_by_dim.restype = st_ctrl_size_t

st_KernelConfig_get_total_num_work_items = \
    sixtracklib.st_KernelConfig_get_total_num_work_items
st_KernelConfig_get_total_num_work_items.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_total_num_work_items.restype = st_ctrl_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_set_num_work_items_1d = \
    sixtracklib.st_KernelConfig_set_num_work_items_1d
st_KernelConfig_set_num_work_items_1d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t]
st_KernelConfig_set_num_work_items_1d.restype = st_ctrl_status_t

st_KernelConfig_set_num_work_items_2d = \
    sixtracklib.st_KernelConfig_set_num_work_items_2d
st_KernelConfig_set_num_work_items_2d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_num_work_items_2d.restype = st_ctrl_status_t

st_KernelConfig_set_num_work_items_3d = \
    sixtracklib.st_KernelConfig_set_num_work_items_3d
st_KernelConfig_set_num_work_items_3d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_num_work_items_3d.restype = st_ctrl_status_t

st_KernelConfig_set_num_work_items = \
    sixtracklib.st_KernelConfig_set_num_work_items
st_KernelConfig_set_num_work_items.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t_p]
st_KernelConfig_set_num_work_items.restype = st_ctrl_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_get_work_item_offset_by_dim = \
    sixtracklib.st_KernelConfig_get_work_item_offset_by_dim
st_KernelConfig_get_work_item_offset_by_dim.argtypes = [
    st_KernelConfigBase_p]
st_KernelConfig_get_work_item_offset_by_dim.restype = st_ctrl_size_t

st_KernelConfig_set_work_item_offset_1d = \
    sixtracklib.st_KernelConfig_set_work_item_offset_1d
st_KernelConfig_set_work_item_offset_1d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t]
st_KernelConfig_set_work_item_offset_1d.restype = st_ctrl_status_t

st_KernelConfig_set_work_item_offset_2d = \
    sixtracklib.st_KernelConfig_set_work_item_offset_2d
st_KernelConfig_set_work_item_offset_2d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_work_item_offset_2d.restype = st_ctrl_status_t

st_KernelConfig_set_work_item_offset_3d = \
    sixtracklib.st_KernelConfig_set_work_item_offset_3d
st_KernelConfig_set_work_item_offset_3d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_work_item_offset_3d.restype = st_ctrl_status_t

st_KernelConfig_set_work_item_offset = \
    sixtracklib.st_KernelConfig_set_work_item_offset
st_KernelConfig_set_work_item_offset.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t_p]
st_KernelConfig_set_work_item_offset.restype = st_ctrl_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_get_work_group_size_by_dim = \
    sixtracklib.st_KernelConfig_get_work_group_size_by_dim
st_KernelConfig_get_work_group_size_by_dim.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_work_group_size_by_dim.restype = st_ctrl_size_t

st_KernelConfig_get_work_groups_dim = \
    sixtracklib.st_KernelConfig_get_work_groups_dim
st_KernelConfig_get_work_groups_dim.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_get_work_groups_dim.restype = st_ctrl_size_t

st_KernelConfig_set_work_group_sizes_1d = \
    sixtracklib.st_KernelConfig_set_work_group_sizes_1d
st_KernelConfig_set_work_group_sizes_1d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t]
st_KernelConfig_set_work_group_sizes_1d.restype = st_ctrl_status_t

st_KernelConfig_set_work_group_sizes_2d = \
    sixtracklib.st_KernelConfig_set_work_group_sizes_2d
st_KernelConfig_set_work_group_sizes_2d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_work_group_sizes_2d.restype = st_ctrl_status_t

st_KernelConfig_set_work_group_sizes_3d = \
    sixtracklib.st_KernelConfig_set_work_group_sizes_3d
st_KernelConfig_set_work_group_sizes_3d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_work_group_sizes_3d.restype = st_ctrl_status_t

st_KernelConfig_set_work_group_sizes = \
    sixtracklib.st_KernelConfig_set_work_group_sizes
st_KernelConfig_set_work_group_sizes.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t_p]
st_KernelConfig_set_work_group_sizes.restype = st_ctrl_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_get_preferred_work_group_multiple_by_dim = \
    sixtracklib.st_KernelConfig_get_preferred_work_group_multiple_by_dim
st_KernelConfig_get_preferred_work_group_multiple_by_dim.argtypes = [
    st_KernelConfigBase_p]
st_KernelConfig_get_preferred_work_group_multiple_by_dim.restype = \
    st_ctrl_size_t

st_KernelConfig_set_preferred_work_group_multiple_1d = \
    sixtracklib.st_KernelConfig_set_preferred_work_group_multiple_1d
st_KernelConfig_set_preferred_work_group_multiple_1d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t]
st_KernelConfig_set_preferred_work_group_multiple_1d.restype = st_ctrl_status_t

st_KernelConfig_set_preferred_work_group_multiple_2d = \
    sixtracklib.st_KernelConfig_set_preferred_work_group_multiple_2d
st_KernelConfig_set_preferred_work_group_multiple_2d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_preferred_work_group_multiple_2d.restype = st_ctrl_status_t

st_KernelConfig_set_preferred_work_group_multiple_3d = \
    sixtracklib.st_KernelConfig_set_preferred_work_group_multiple_3d
st_KernelConfig_set_preferred_work_group_multiple_3d.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_set_preferred_work_group_multiple_3d.restype = st_ctrl_status_t

st_KernelConfig_set_preferred_work_group_multiple = \
    sixtracklib.st_KernelConfig_set_preferred_work_group_multiple
st_KernelConfig_set_preferred_work_group_multiple.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t_p]
st_KernelConfig_set_preferred_work_group_multiple.restype = st_ctrl_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_clear = sixtracklib.st_KernelConfig_clear
st_KernelConfig_clear.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_clear.restype = None

st_KernelConfig_reset = sixtracklib.st_KernelConfig_reset
st_KernelConfig_reset.argtypes = [
    st_KernelConfigBase_p, st_ctrl_size_t, st_ctrl_size_t]
st_KernelConfig_reset.restype = None

st_KernelConfig_needs_update = sixtracklib.st_KernelConfig_needs_update
st_KernelConfig_needs_update.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_needs_update.restype = st_ctrl_status_t

st_KernelConfig_update = sixtracklib.st_KernelConfig_update
st_KernelConfig_update.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_update.restype = st_ctrl_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_KernelConfig_print_out = sixtracklib.st_KernelConfig_print_out
st_KernelConfig_print_out.argtypes = [st_KernelConfigBase_p]
st_KernelConfig_print_out.restype = None

# -----------------------------------------------------------------------------
# NS(ControllerBase), NS(NodeControllerBase)

st_ControllerBase_p = ct.c_void_p
st_NullControllerBase = ct.cast(0, st_ControllerBase_p)

st_ArgumentBase_p = ct.c_void_p
st_NullArgumentBase = ct.cast(0, st_ArgumentBase_p)

st_Controller_delete = sixtracklib.st_Controller_delete
st_Controller_delete.argtypes = [st_ControllerBase_p]
st_Controller_delete.restype = None

st_Controller_clear = sixtracklib.st_Controller_clear
st_Controller_clear.argtypes = [st_ControllerBase_p]
st_Controller_clear.restype = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_arch_id = sixtracklib.st_Controller_get_arch_id
st_Controller_get_arch_id.argtypes = [st_ControllerBase_p]
st_Controller_get_arch_id.restype = st_arch_id_t

st_Controller_has_arch_string = sixtracklib.st_Controller_has_arch_string
st_Controller_has_arch_string.argtypes = [st_ControllerBase_p]
st_Controller_has_arch_string.restype = ct.c_bool

st_Controller_get_arch_string = sixtracklib.st_Controller_get_arch_string
st_Controller_get_arch_string.argtypes = [st_ControllerBase_p]
st_Controller_get_arch_string.restype = ct.c_char_p

st_Controller_has_config_string = sixtracklib.st_Controller_has_config_string
st_Controller_has_config_string.argtypes = [st_ControllerBase_p]
st_Controller_has_config_string.restype = ct.c_bool

st_Controller_get_config_string = sixtracklib.st_Controller_get_config_string
st_Controller_get_config_string.argtypes = [st_ControllerBase_p]
st_Controller_get_config_string.restype = ct.c_char_p

st_Controller_uses_nodes = sixtracklib.st_Controller_uses_nodes
st_Controller_uses_nodes.argtypes = [st_ControllerBase_p]
st_Controller_uses_nodes.restype = ct.c_bool

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_send_detailed = sixtracklib.st_Controller_send_detailed
st_Controller_send_detailed.argtypes = [
    st_ControllerBase_p,
    st_ArgumentBase_p,
    ct.c_void_p,
    st_arch_size_t]
st_Controller_send_detailed.restype = st_arch_status_t

st_Controller_send_buffer = sixtracklib.st_Controller_send_buffer
st_Controller_send_buffer.argtypes = [st_ControllerBase_p,
                                      st_ArgumentBase_p, st_Buffer_p]
st_Controller_send_buffer.restype = st_arch_status_t

st_Controller_receive_detailed = sixtracklib.st_Controller_receive_detailed
st_Controller_receive_detailed.argtypes = [
    st_ControllerBase_p,
    ct.c_void_p,
    st_arch_size_t,
    st_ArgumentBase_p]
st_Controller_receive_detailed.restype = st_arch_status_t

st_Controller_receive_buffer = sixtracklib.st_Controller_receive_buffer
st_Controller_receive_buffer.argtypes = [st_ControllerBase_p, st_Buffer_p,
                                         st_ArgumentBase_p]
st_Controller_receive_buffer.restype = st_arch_status_t

st_Controller_is_cobjects_buffer_arg_remapped = \
    sixtracklib.st_Controller_is_cobjects_buffer_arg_remapped
st_Controller_is_cobjects_buffer_arg_remapped.argtypes = [st_ControllerBase_p,
                                                          st_ArgumentBase_p]
st_Controller_is_cobjects_buffer_arg_remapped.restype = ct.c_bool

st_Controller_remap_cobjects_buffer_arg = \
    sixtracklib.st_Controller_remap_cobjects_buffer_arg
st_Controller_remap_cobjects_buffer_arg.argtypes = [st_ControllerBase_p,
                                                    st_ArgumentBase_p]
st_Controller_remap_cobjects_buffer_arg.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_is_ready_to_run_kernel = \
    sixtracklib.st_Controller_is_ready_to_run_kernel
st_Controller_is_ready_to_run_kernel.argtypes = [st_ControllerBase_p]
st_Controller_is_ready_to_run_kernel.restype = ct.c_bool

st_Controller_is_ready_to_remap = sixtracklib.st_Controller_is_ready_to_remap
st_Controller_is_ready_to_remap.argtypes = [st_ControllerBase_p]
st_Controller_is_ready_to_remap.restype = ct.c_bool

st_Controller_is_ready_to_send = sixtracklib.st_Controller_is_ready_to_send
st_Controller_is_ready_to_send.argtypes = [st_ControllerBase_p]
st_Controller_is_ready_to_send.restype = ct.c_bool

st_Controller_is_ready_to_receive = \
    sixtracklib.st_Controller_is_ready_to_receive
st_Controller_is_ready_to_receive.argtypes = [st_ControllerBase_p]
st_Controller_is_ready_to_receive.restype = ct.c_bool

st_Controller_is_in_debug_mode = \
    sixtracklib.st_Controller_is_in_debug_mode
st_Controller_is_in_debug_mode.argtypes = [st_ControllerBase_p]
st_Controller_is_in_debug_mode.restype = ct.c_bool

st_Controller_enable_debug_mode = sixtracklib.st_Controller_enable_debug_mode
st_Controller_enable_debug_mode.argtypes = [st_ControllerBase_p]
st_Controller_enable_debug_mode.restype = st_arch_status_t

st_Controller_disable_debug_mode = sixtracklib.st_Controller_disable_debug_mode
st_Controller_disable_debug_mode.argtypes = [st_ControllerBase_p]
st_Controller_disable_debug_mode.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_num_of_kernels = sixtracklib.st_Controller_get_num_of_kernels
st_Controller_get_num_of_kernels.argtypes = [st_ControllerBase_p]
st_Controller_get_num_of_kernels.restype = st_arch_size_t

st_Controller_get_kernel_work_items_dim = \
    sixtracklib.st_Controller_get_kernel_work_items_dim
st_Controller_get_kernel_work_items_dim.argtypes = [st_ControllerBase_p,
                                                    st_kernel_id_t]
st_Controller_get_kernel_work_items_dim.restype = st_arch_size_t

st_Controller_get_kernel_work_groups_dim = \
    sixtracklib. st_Controller_get_kernel_work_groups_dim
st_Controller_get_kernel_work_groups_dim.argtypes = [st_ControllerBase_p,
                                                     st_kernel_id_t]
st_Controller_get_kernel_work_groups_dim.restype = st_arch_size_t

st_Controller_get_num_of_kernel_arguments = \
    sixtracklib.st_Controller_get_num_of_kernel_arguments
st_Controller_get_num_of_kernel_arguments.argtypes = [st_ControllerBase_p,
                                                      st_kernel_id_t]
st_Controller_get_num_of_kernel_arguments.restype = st_arch_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_kernel_has_name = sixtracklib.st_Controller_kernel_has_name
st_Controller_kernel_has_name.argtypes = [st_ControllerBase_p,
                                          st_kernel_id_t]
st_Controller_kernel_has_name.restype = ct.c_bool

st_Controller_get_kernel_name_string = \
    sixtracklib.st_Controller_get_kernel_name_string
st_Controller_get_kernel_name_string.argtypes = [st_ControllerBase_p,
                                                 st_kernel_id_t]
st_Controller_get_kernel_name_string.restype = ct.c_char_p

st_Controller_has_kernel_id = sixtracklib.st_Controller_has_kernel_id
st_Controller_has_kernel_id.argtypes = [st_ControllerBase_p,
                                        st_kernel_id_t]
st_Controller_has_kernel_id.restype = ct.c_bool

st_Controller_has_kernel_by_name = sixtracklib.st_Controller_has_kernel_by_name
st_Controller_has_kernel_by_name.argtypes = [st_ControllerBase_p,
                                             ct.c_char_p]
st_Controller_has_kernel_by_name.restype = ct.c_bool

st_Controller_get_ptr_kernel_config_base = \
    sixtracklib.st_Controller_get_ptr_kernel_config_base
st_Controller_get_ptr_kernel_config_base.argtypes = [st_ControllerBase_p,
                                                     st_kernel_id_t]
st_Controller_get_ptr_kernel_config_base.restype = st_KernelConfigBase_p

st_Controller_get_ptr_kernel_config_base_by_name = \
    sixtracklib.st_Controller_get_ptr_kernel_config_base_by_name
st_Controller_get_ptr_kernel_config_base_by_name.argtypes = [
    st_ControllerBase_p, ct.c_char_p]
st_Controller_get_ptr_kernel_config_base_by_name.restype = st_KernelConfigBase_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_has_remap_cobject_buffer_kernel = \
    sixtracklib.st_Controller_has_remap_cobject_buffer_kernel
st_Controller_has_remap_cobject_buffer_kernel.argtypes = [st_ControllerBase_p]
st_Controller_has_remap_cobject_buffer_kernel.restype = ct.c_bool

st_Controller_get_remap_cobject_buffer_kernel_id = \
    sixtracklib.st_Controller_get_remap_cobject_buffer_kernel_id
st_Controller_get_remap_cobject_buffer_kernel_id.argtypes = [
    st_ControllerBase_p]
st_Controller_get_remap_cobject_buffer_kernel_id.restype = st_kernel_id_t


st_Controller_set_remap_cobject_buffer_kernel_id = \
    sixtracklib.st_Controller_set_remap_cobject_buffer_kernel_id
st_Controller_set_remap_cobject_buffer_kernel_id.argtypes = [
    st_ControllerBase_p, st_kernel_id_t]
st_Controller_set_remap_cobject_buffer_kernel_id.restype = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_has_remap_cobject_buffer_debug_kernel = \
    sixtracklib.st_Controller_has_remap_cobject_buffer_debug_kernel
st_Controller_has_remap_cobject_buffer_debug_kernel.argtypes = [
    st_ControllerBase_p]
st_Controller_has_remap_cobject_buffer_debug_kernel.restype = ct.c_bool

st_Controller_get_remap_cobject_buffer_debug_kernel_id = \
    sixtracklib.st_Controller_get_remap_cobject_buffer_debug_kernel_id
st_Controller_get_remap_cobject_buffer_debug_kernel_id.argtypes = [
    st_ControllerBase_p]
st_Controller_get_remap_cobject_buffer_debug_kernel_id.restype = st_kernel_id_t

st_Controller_set_remap_cobject_buffer_debug_kernel_id = \
    sixtracklib.st_Controller_set_remap_cobject_buffer_debug_kernel_id
st_Controller_set_remap_cobject_buffer_debug_kernel_id.argtypes = [
    st_ControllerBase_p, st_kernel_id_t]
st_Controller_set_remap_cobject_buffer_debug_kernel_id.restype = None

# ------------------------------------------------------------------------------
# NS(NodeControllerBase):

st_NODE_UNDEFINED_INDEX = st_node_index_t(int('0xFFFFFFFF', 16))
st_NODE_ILLEGAL_PLATFORM_ID = st_node_platform_id_t(-1)
st_NODE_ILLEGAL_DEVICE_ID = st_node_device_id_t(-1)

st_Controller_get_num_available_nodes = \
    sixtracklib.st_Controller_get_num_available_nodes
st_Controller_get_num_available_nodes.argtypes = [st_ControllerBase_p]
st_Controller_get_num_available_nodes.restype = st_node_index_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_has_default_node = sixtracklib.st_Controller_has_default_node
st_Controller_has_default_node.argtypes = [st_ControllerBase_p]
st_Controller_has_default_node.restype = ct.c_bool

st_Controller_get_default_node_index = \
    sixtracklib.st_Controller_get_default_node_index
st_Controller_get_default_node_index.argtypes = [st_ControllerBase_p]
st_Controller_get_default_node_index.restype = st_node_index_t

st_Controller_get_default_node_id = \
    sixtracklib.st_Controller_get_default_node_id
st_Controller_get_default_node_id.argtypes = [st_ControllerBase_p]
st_Controller_get_default_node_id.restype = st_NodeId_p

st_Controller_get_default_node_info_base = \
    sixtracklib.st_Controller_get_default_node_info_base
st_Controller_get_default_node_info_base.argtypes = [st_ControllerBase_p]
st_Controller_get_default_node_info_base.restype = st_NodeInfoBase_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_is_default_node_index = \
    sixtracklib.st_Controller_is_default_node_index
st_Controller_is_default_node_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_is_default_node_index.restype = ct.c_bool

st_Controller_is_default_node_id = \
    sixtracklib.st_Controller_is_default_node_id
st_Controller_is_default_node_id.argtypes = [st_ControllerBase_p, st_NodeId_p]
st_Controller_is_default_node_id.restype = ct.c_bool

st_Controller_is_default_platform_id_and_device_id = \
    sixtracklib.st_Controller_is_default_platform_id_and_device_id
st_Controller_is_default_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_is_default_platform_id_and_device_id.restype = ct.c_bool

st_Controller_is_default_node = sixtracklib.st_Controller_is_default_node
st_Controller_is_default_node.argtypes = [st_ControllerBase_p, ct.c_char_p]
st_Controller_is_default_node.restype = ct.c_bool

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_is_node_available_by_index = \
    sixtracklib.st_Controller_is_node_available_by_index
st_Controller_is_node_available_by_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_is_node_available_by_index.restype = ct.c_bool

st_Controller_is_node_available_by_node_id = \
    sixtracklib.st_Controller_is_node_available_by_node_id
st_Controller_is_node_available_by_node_id.argtypes = [
    st_ControllerBase_p, st_NodeId_p]
st_Controller_is_node_available_by_node_id.restype = ct.c_bool

st_Controller_is_node_available_by_platform_id_and_device_id = \
    sixtracklib.st_Controller_is_node_available_by_platform_id_and_device_id
st_Controller_is_node_available_by_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_is_node_available_by_platform_id_and_device_id.restype = ct.c_bool

st_Controller_is_node_available = \
    sixtracklib.st_Controller_is_node_available
st_Controller_is_node_available.argtypes = [st_ControllerBase_p, ct.c_char_p]
st_Controller_is_node_available.restype = ct.c_bool

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_min_available_node_index = \
    sixtracklib.st_Controller_get_min_available_node_index
st_Controller_get_min_available_node_index.argtypes = [st_ControllerBase_p]
st_Controller_get_min_available_node_index.restype = st_node_index_t

st_Controller_get_max_available_node_index = \
    sixtracklib.st_Controller_get_max_available_node_index
st_Controller_get_max_available_node_index.argtypes = [st_ControllerBase_p]
st_Controller_get_max_available_node_index.restype = st_node_index_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_available_node_indices = \
    sixtracklib.st_Controller_get_available_node_indices
st_Controller_get_available_node_indices.argtypes = [
    st_ControllerBase_p, st_ctrl_size_t, st_ctrl_size_t_p]
st_Controller_get_available_node_indices.restype = st_ctrl_size_t

st_Controller_get_available_node_ids = \
    sixtracklib.st_Controller_get_available_node_ids
st_Controller_get_available_node_ids.argtypes = [
    st_ControllerBase_p, st_ctrl_size_t, st_NodeId_p]
st_Controller_get_available_node_ids.restype = st_ctrl_size_t

st_Controller_get_available_base_node_infos = \
    sixtracklib.st_Controller_get_available_base_node_infos
st_Controller_get_available_base_node_infos.argtypes = [
    st_ControllerBase_p, st_ctrl_size_t, ct.POINTER(st_NodeInfoBase_p)]
st_Controller_get_available_base_node_infos.restype = st_ctrl_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_node_index_by_node_id = \
    sixtracklib.st_Controller_get_node_index_by_node_id
st_Controller_get_node_index_by_node_id.argtypes = [
    st_ControllerBase_p, st_NodeId_p]
st_Controller_get_node_index_by_node_id.restype = st_node_index_t

st_Controller_get_node_index_by_platform_id_and_device_id =  \
    sixtracklib.st_Controller_get_node_index_by_platform_id_and_device_id
st_Controller_get_node_index_by_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_get_node_index_by_platform_id_and_device_id.restype = \
    st_node_index_t

st_Controller_get_node_index_by_node_info = \
    sixtracklib.st_Controller_get_node_index_by_node_info
st_Controller_get_node_index_by_node_info.argtypes = [
    st_ControllerBase_p, st_NodeInfoBase_p]
st_Controller_get_node_index_by_node_info.restype = st_node_index_t

st_Controller_get_node_index = sixtracklib.st_Controller_get_node_index
st_Controller_get_node_index.argtypes = [st_ControllerBase_p, ct.c_char_p]
st_Controller_get_node_index.restype = st_node_index_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_ptr_node_id_by_index = \
    sixtracklib.st_Controller_get_ptr_node_id_by_index
st_Controller_get_ptr_node_id_by_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_get_ptr_node_id_by_index.restype = st_NodeId_p

st_Controller_get_ptr_node_id_by_platform_id_and_device_id = \
    sixtracklib.st_Controller_get_ptr_node_id_by_platform_id_and_device_id
st_Controller_get_ptr_node_id_by_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_get_ptr_node_id_by_platform_id_and_device_id.restype = st_NodeId_p

st_Controller_get_ptr_node_id = sixtracklib.st_Controller_get_ptr_node_id
st_Controller_get_ptr_node_id.argtypes = [st_ControllerBase_p, ct.c_char_p]
st_Controller_get_ptr_node_id.restype = st_NodeId_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_get_ptr_node_info_base_by_index = \
    sixtracklib.st_Controller_get_ptr_node_info_base_by_index
st_Controller_get_ptr_node_info_base_by_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_get_ptr_node_info_base_by_index.restype = st_NodeInfoBase_p

st_Controller_get_ptr_node_info_base_by_node_id = \
    sixtracklib.st_Controller_get_ptr_node_info_base_by_node_id
st_Controller_get_ptr_node_info_base_by_node_id.argtypes = [
    st_ControllerBase_p, st_NodeId_p]
st_Controller_get_ptr_node_info_base_by_node_id.restype = st_NodeInfoBase_p

st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id = \
    sixtracklib.st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id
st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_get_ptr_node_info_base_by_platform_id_and_device_id.restype = \
    st_NodeInfoBase_p

st_Controller_get_ptr_node_info_base = \
    sixtracklib.st_Controller_get_ptr_node_info_base
st_Controller_get_ptr_node_info_base.argtypes = [
    st_ControllerBase_p, ct.c_char_p]
st_Controller_get_ptr_node_info_base.restype = st_NodeInfoBase_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_has_selected_node = sixtracklib.st_Controller_has_selected_node
st_Controller_has_selected_node.argtypes = [st_ControllerBase_p]
st_Controller_has_selected_node.restype = ct.c_bool

st_Controller_get_selected_node_index = \
    sixtracklib.st_Controller_get_selected_node_index
st_Controller_get_selected_node_index.argtypes = [st_ControllerBase_p]
st_Controller_get_selected_node_index.restype = st_node_index_t

st_Controller_get_ptr_selected_node_id = \
    sixtracklib.st_Controller_get_ptr_selected_node_id
st_Controller_get_ptr_selected_node_id.argtypes = [st_ControllerBase_p]
st_Controller_get_ptr_selected_node_id.restype = st_NodeId_p

st_Controller_get_ptr_selected_node_info_base = \
    sixtracklib.st_Controller_get_ptr_selected_node_info_base
st_Controller_get_ptr_selected_node_info_base.argtypes = [st_ControllerBase_p]
st_Controller_get_ptr_selected_node_info_base.restype = st_NodeInfoBase_p

st_Controller_get_selected_node_id_str = \
    sixtracklib.st_Controller_get_selected_node_id_str
st_Controller_get_selected_node_id_str.argtypes = [st_ControllerBase_p]
st_Controller_get_selected_node_id_str.restype = ct.c_char_p

_st_Controller_copy_selected_node_id_str = \
    sixtracklib.st_Controller_copy_selected_node_id_str
_st_Controller_copy_selected_node_id_str.argtypes = [
    st_ControllerBase_p, ct.c_char_p, st_arch_size_t]
_st_Controller_copy_selected_node_id_str.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_select_node = sixtracklib.st_Controller_select_node
st_Controller_select_node.argtypes = [st_ControllerBase_p, ct.c_char_p]
st_Controller_select_node.restype = st_arch_status_t

st_Controller_select_node_by_node_id = \
    sixtracklib.st_Controller_select_node_by_node_id
st_Controller_select_node_by_node_id.argtypes = [
    st_ControllerBase_p, st_NodeId_p]
st_Controller_select_node_by_node_id.rstype = st_arch_status_t

st_Controller_select_node_by_plaform_id_and_device_id = \
    sixtracklib.st_Controller_select_node_by_plaform_id_and_device_id
st_Controller_select_node_by_plaform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_select_node_by_plaform_id_and_device_id.restype = \
    st_arch_status_t

st_Controller_select_node_by_index = \
    sixtracklib.st_Controller_select_node_by_index
st_Controller_select_node_by_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_select_node_by_index.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_can_change_selected_node = \
    sixtracklib.st_Controller_can_change_selected_node
st_Controller_can_change_selected_node.argtypes = [
    st_ControllerBase_p]
st_Controller_can_change_selected_node.restype = ct.c_bool

st_Controller_can_directly_change_selected_node = \
    sixtracklib.st_Controller_can_directly_change_selected_node
st_Controller_can_directly_change_selected_node.argtypes = [
    st_ControllerBase_p]
st_Controller_can_directly_change_selected_node.restype = ct.c_bool

st_Controller_change_selected_node = \
    sixtracklib.st_Controller_change_selected_node
st_Controller_change_selected_node.argtypes = [
    st_ControllerBase_p, st_node_index_t, st_node_index_t]
st_Controller_change_selected_node.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_can_unselect_node = sixtracklib.st_Controller_can_unselect_node
st_Controller_can_unselect_node.argtypes = [st_ControllerBase_p]
st_Controller_can_unselect_node.restype = ct.c_bool

st_Controller_unselect_node = sixtracklib.st_Controller_unselect_node
st_Controller_unselect_node.argtypes = [st_ControllerBase_p]
st_Controller_unselect_node.restype = st_arch_status_t

st_Controller_unselect_node_by_index = \
    sixtracklib.st_Controller_unselect_node_by_index
st_Controller_unselect_node_by_index.argtypes = [
    st_ControllerBase_p, st_node_index_t]
st_Controller_unselect_node_by_index.restype = st_arch_status_t

st_Controller_unselect_node_by_node_id = \
    sixtracklib.st_Controller_unselect_node_by_node_id
st_Controller_unselect_node_by_node_id.argtypes = [
    st_ControllerBase_p, st_NodeId_p]
st_Controller_unselect_node_by_node_id.restype = st_arch_status_t

st_Controller_unselect_node_by_platform_id_and_device_id = \
    sixtracklib.st_Controller_unselect_node_by_platform_id_and_device_id
st_Controller_unselect_node_by_platform_id_and_device_id.argtypes = [
    st_ControllerBase_p, st_node_platform_id_t, st_node_device_id_t]
st_Controller_unselect_node_by_platform_id_and_device_id.restype = \
    st_arch_status_t

st_Controller_unselect_node_by_node_id_str = \
    sixtracklib.st_Controller_unselect_node_by_node_id_str
st_Controller_unselect_node_by_node_id_str.argtypes = [
    st_ControllerBase_p, ct.c_char_p]
st_Controller_unselect_node_by_node_id_str.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Controller_print_out_available_nodes_info = \
    sixtracklib.st_Controller_print_out_available_nodes_info
st_Controller_print_out_available_nodes_info.argtypes = [
    st_ControllerBase_p]
st_Controller_print_out_available_nodes_info.restype = None

st_Controller_store_available_nodes_info_to_string = \
    sixtracklib.st_Controller_store_available_nodes_info_to_string
st_Controller_store_available_nodes_info_to_string.argtypes = [
    st_ControllerBase_p, ct.c_char_p, st_arch_size_t, st_arch_size_t_p]
st_Controller_store_available_nodes_info_to_string.restype = None

# -----------------------------------------------------------------------------
# NS(ArgumentBase):

st_Argument_delete = sixtracklib.st_Argument_delete
st_Argument_delete.argtypes = [st_ArgumentBase_p]
st_Argument_delete.restype = None

st_Argument_get_arch_id = sixtracklib.st_Argument_get_arch_id
st_Argument_get_arch_id.argtypes = [st_ArgumentBase_p]
st_Argument_get_arch_id.restype = st_arch_id_t

st_Argument_has_arch_string = sixtracklib.st_Argument_has_arch_string
st_Argument_has_arch_string.argtypes = [st_ArgumentBase_p]
st_Argument_has_arch_string.restype = ct.c_bool

st_Argument_get_arch_string = sixtracklib.st_Argument_get_arch_string
st_Argument_get_arch_string.argtypes = [st_ArgumentBase_p]
st_Argument_get_arch_string.restype = ct.c_char_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_send_again = sixtracklib.st_Argument_send_again
st_Argument_send_again.argtypes = [st_ArgumentBase_p]
st_Argument_send_again.restype = st_arch_status_t

st_Argument_send_buffer = sixtracklib.st_Argument_send_buffer
st_Argument_send_buffer.argtypes = [st_ArgumentBase_p, st_Buffer_p]
st_Argument_send_buffer.restype = st_arch_status_t

st_Argument_send_buffer_without_remap = \
    sixtracklib.st_Argument_send_buffer_without_remap
st_Argument_send_buffer_without_remap.argtypes = [
    st_ArgumentBase_p, st_Buffer_p]
st_Argument_send_buffer_without_remap.restype = st_arch_status_t

st_Argument_send_raw_argument = sixtracklib.st_Argument_send_raw_argument
st_Argument_send_raw_argument.argtypes = [
    st_ArgumentBase_p, ct.c_void_p, st_arch_size_t]
st_Argument_send_raw_argument.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_receive_again = sixtracklib.st_Argument_receive_again
st_Argument_receive_again.argtypes = [st_ArgumentBase_p]
st_Argument_receive_again.restype = st_arch_status_t

st_Argument_receive_buffer = sixtracklib.st_Argument_receive_buffer
st_Argument_receive_buffer.argtypes = [st_ArgumentBase_p, st_Buffer_p]
st_Argument_receive_buffer.restype = st_arch_status_t

st_Argument_receive_buffer_without_remap = \
    sixtracklib.st_Argument_receive_buffer_without_remap
st_Argument_receive_buffer_without_remap.argtypes = [
    st_ArgumentBase_p, st_Buffer_p]
st_Argument_receive_buffer_without_remap.restype = st_arch_status_t

st_Argument_receive_raw_argument = sixtracklib.st_Argument_receive_raw_argument
st_Argument_receive_raw_argument.argtypes = [
    st_ArgumentBase_p, ct.c_void_p, st_arch_size_t]
st_Argument_receive_raw_argument.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_remap_cobjects_buffer = \
    sixtracklib.st_Argument_remap_cobjects_buffer
st_Argument_remap_cobjects_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_remap_cobjects_buffer.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_uses_cobjects_buffer = \
    sixtracklib.st_Argument_uses_cobjects_buffer
st_Argument_uses_cobjects_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_uses_cobjects_buffer.restype = ct.c_bool

st_Argument_get_const_cobjects_buffer = \
    sixtracklib.st_Argument_get_const_cobjects_buffer
st_Argument_get_const_cobjects_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_get_const_cobjects_buffer.restype = st_Buffer_p

st_Argument_get_cobjects_buffer = sixtracklib.st_Argument_get_cobjects_buffer
st_Argument_get_cobjects_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_get_cobjects_buffer.restype = st_Buffer_p

st_Argument_get_cobjects_buffer_slot_size = \
    sixtracklib.st_Argument_get_cobjects_buffer_slot_size
st_Argument_get_cobjects_buffer_slot_size.argtypes = [st_ArgumentBase_p]
st_Argument_get_cobjects_buffer_slot_size.restype = st_buffer_size_t


st_Argument_uses_raw_argument = sixtracklib.st_Argument_uses_raw_argument
st_Argument_uses_raw_argument.argtypes = [st_ArgumentBase_p]
st_Argument_uses_raw_argument.restype = ct.c_bool

st_Argument_get_const_ptr_raw_argument = \
    sixtracklib.st_Argument_get_const_ptr_raw_argument
st_Argument_get_const_ptr_raw_argument.argtypes = [st_ArgumentBase_p]
st_Argument_get_const_ptr_raw_argument.restype = ct.c_void_p

st_Argument_get_ptr_raw_argument = \
    sixtracklib.st_Argument_get_ptr_raw_argument
st_Argument_get_ptr_raw_argument.argtypes = [st_ArgumentBase_p]
st_Argument_get_ptr_raw_argument.restype = ct.c_void_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_get_size = sixtracklib.st_Argument_get_size
st_Argument_get_size.argtypes = [st_ArgumentBase_p]
st_Argument_get_size.restype = st_arch_size_t

st_Argument_get_capacity = sixtracklib.st_Argument_get_capacity
st_Argument_get_capacity.argtypes = [st_ArgumentBase_p]
st_Argument_get_capacity.restype = st_arch_size_t

st_Argument_has_argument_buffer = sixtracklib.st_Argument_has_argument_buffer
st_Argument_has_argument_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_has_argument_buffer.restype = ct.c_bool

st_Argument_requires_argument_buffer = \
    sixtracklib.st_Argument_has_argument_buffer
st_Argument_has_argument_buffer.argtypes = [st_ArgumentBase_p]
st_Argument_has_argument_buffer.restype = ct.c_bool

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_Argument_get_ptr_base_controller = \
    sixtracklib.st_Argument_get_ptr_base_controller
st_Argument_get_ptr_base_controller.argtypes = [st_ArgumentBase_p]
st_Argument_get_ptr_base_controller.restype = st_ControllerBase_p

st_Argument_get_const_ptr_base_controller = \
    sixtracklib.st_Argument_get_const_ptr_base_controller
st_Argument_get_const_ptr_base_controller.argtypes = [st_ArgumentBase_p]
st_Argument_get_const_ptr_base_controller.restype = st_ControllerBase_p

# -----------------------------------------------------------------------------
# NS(TrackJobBase):

st_TrackJobBaseNew_p = ct.c_void_p
st_NullTrackJobBaseNew = ct.cast(0, st_TrackJobBaseNew_p)

st_track_status_t = ct.c_int32
st_track_job_collect_flag_t = ct.c_uint16
st_track_job_clear_flag_t = ct.c_uint16
st_track_job_size_t = ct.c_uint64

st_TRACK_SUCCESS = st_track_status_t(0)
st_TRACK_STATUS_GENERAL_FAILURE = st_track_status_t(-1)

st_TrackJobNew_create = sixtracklib.st_TrackJobNew_create
st_TrackJobNew_create.argtypes = [ct.c_char_p, ct.c_char_p]
st_TrackJobNew_create.restype = st_TrackJobBaseNew_p

st_TrackJobNew_new = sixtracklib.st_TrackJobNew_new
st_TrackJobNew_new.argtypes = [
    ct.c_char_p, st_Buffer_p, st_Buffer_p, ct.c_char_p]
st_TrackJobNew_new.restype = st_TrackJobBaseNew_p

st_TrackJobNew_new_with_output = sixtracklib.st_TrackJobNew_new_with_output
st_TrackJobNew_new_with_output.argtypes = [
    ct.c_char_p,
    st_Buffer_p,
    st_Buffer_p,
    st_Buffer_p,
    st_buffer_size_t,
    ct.c_char_p]
st_TrackJobNew_new_with_output.restype = st_TrackJobBaseNew_p

st_TrackJobNew_new_detailed = sixtracklib.st_TrackJobNew_new_detailed
st_TrackJobNew_new_detailed.argtypes = [
    ct.c_char_p,
    st_Buffer_p,
    st_buffer_size_t,
    st_buffer_size_p,
    st_Buffer_p,
    st_Buffer_p,
    st_buffer_size_t,
    ct.c_char_p]
st_TrackJobNew_new_detailed.restype = st_TrackJobBaseNew_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_delete = sixtracklib.st_TrackJobNew_delete
st_TrackJobNew_delete.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_delete.restype = None

st_TrackJobNew_track_until = sixtracklib.st_TrackJobNew_track_until
st_TrackJobNew_track_until.argtypes = [st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_track_until.restype = st_track_status_t

st_TrackJobNew_track_elem_by_elem = \
    sixtracklib.st_TrackJobNew_track_elem_by_elem
st_TrackJobNew_track_elem_by_elem.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_track_elem_by_elem.restype = st_track_status_t

st_TrackJobNew_track_line = sixtracklib.st_TrackJobNew_track_line
st_TrackJobNew_track_line.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t, st_buffer_size_t, ct.c_bool]
st_TrackJobNew_track_line.restype = st_track_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_collect = sixtracklib.st_TrackJobNew_collect
st_TrackJobNew_collect.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect.restype = st_track_job_collect_flag_t

st_TrackJobNew_collect_detailed = sixtracklib.st_TrackJobNew_collect_detailed
st_TrackJobNew_collect_detailed.argtypes = [
    st_TrackJobBaseNew_p, st_track_job_collect_flag_t]
st_TrackJobNew_collect_detailed.restype = st_track_job_collect_flag_t

st_TrackJobNew_collect_particles = sixtracklib.st_TrackJobNew_collect_particles
st_TrackJobNew_collect_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect_particles.restype = st_arch_status_t

st_TrackJobNew_collect_beam_elements = \
    sixtracklib.st_TrackJobNew_collect_beam_elements
st_TrackJobNew_collect_beam_elements.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect_beam_elements.restype = st_arch_status_t

st_TrackJobNew_collect_output = sixtracklib.st_TrackJobNew_collect_output
st_TrackJobNew_collect_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect_output.restype = st_arch_status_t

st_TrackJobNew_collect_debug_flag = \
    sixtracklib.st_TrackJobNew_collect_debug_flag
st_TrackJobNew_collect_debug_flag.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect_debug_flag.restype = st_arch_status_t

st_TrackJobNew_collect_particles_addresses = \
    sixtracklib.st_TrackJobNew_collect_particles_addresses
st_TrackJobNew_collect_particles_addresses.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_collect_particles_addresses.restype = st_arch_status_t

st_TrackJobNew_enable_collect_particles = \
    sixtracklib.st_TrackJobNew_enable_collect_particles
st_TrackJobNew_enable_collect_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_enable_collect_particles.restype = None

st_TrackJobNew_disable_collect_particles = \
    sixtracklib.st_TrackJobNew_disable_collect_particles
st_TrackJobNew_disable_collect_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_disable_collect_particles.restype = None

st_TrackJobNew_is_collecting_particles = \
    sixtracklib.st_TrackJobNew_is_collecting_particles
st_TrackJobNew_is_collecting_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_is_collecting_particles.restype = ct.c_bool

st_TrackJobNew_enable_collect_beam_elements = \
    sixtracklib.st_TrackJobNew_enable_collect_beam_elements
st_TrackJobNew_enable_collect_beam_elements.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_enable_collect_beam_elements.restype = None

st_TrackJobNew_disable_collect_beam_elements = \
    sixtracklib.st_TrackJobNew_disable_collect_beam_elements
st_TrackJobNew_disable_collect_beam_elements.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_disable_collect_beam_elements.restype = None

st_TrackJobNew_is_collecting_beam_elements = \
    sixtracklib.st_TrackJobNew_is_collecting_beam_elements
st_TrackJobNew_is_collecting_beam_elements.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_is_collecting_beam_elements.restype = ct.c_bool

st_TrackJobNew_enable_collect_output = \
    sixtracklib.st_TrackJobNew_enable_collect_output
st_TrackJobNew_enable_collect_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_enable_collect_output.restype = None

st_TrackJobNew_disable_collect_output = \
    sixtracklib.st_TrackJobNew_disable_collect_output
st_TrackJobNew_disable_collect_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_disable_collect_output.restype = None

st_TrackJobNew_is_collecting_output = \
    sixtracklib.st_TrackJobNew_is_collecting_output
st_TrackJobNew_is_collecting_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_is_collecting_output.restype = ct.c_bool

st_TrackJobNew_get_collect_flags = sixtracklib.st_TrackJobNew_get_collect_flags
st_TrackJobNew_get_collect_flags.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_collect_flags.restype = st_track_job_collect_flag_t

st_TrackJobNew_set_collect_flags = sixtracklib.st_TrackJobNew_set_collect_flags
st_TrackJobNew_set_collect_flags.argtypes = [
    st_TrackJobBaseNew_p, st_track_job_collect_flag_t]
st_TrackJobNew_set_collect_flags.restype = None

st_TrackJobNew_requires_collecting = \
    sixtracklib.st_TrackJobNew_requires_collecting
st_TrackJobNew_requires_collecting.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_requires_collecting.restype = ct.c_bool

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_push = sixtracklib.st_TrackJobNew_push
st_TrackJobNew_push.argtypes = [st_TrackJobBaseNew_p, ct.c_uint16]
st_TrackJobNew_push.restype = ct.c_uint16

st_TrackJobNew_push_particles = sixtracklib.st_TrackJobNew_push_particles
st_TrackJobNew_push_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_push_particles.restype = st_arch_status_t

st_TrackJobNew_push_beam_elements = \
    sixtracklib.st_TrackJobNew_push_beam_elements
st_TrackJobNew_push_beam_elements.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_push_beam_elements.restype = st_arch_status_t

st_TrackJobNew_push_output = sixtracklib.st_TrackJobNew_push_output
st_TrackJobNew_push_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_push_output.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_can_fetch_particle_addresses = \
    sixtracklib.st_TrackJobNew_can_fetch_particle_addresses
st_TrackJobNew_can_fetch_particle_addresses.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_can_fetch_particle_addresses.restype = ct.c_bool

st_TrackJobNew_has_particle_addresses = \
    sixtracklib.st_TrackJobNew_has_particle_addresses
st_TrackJobNew_has_particle_addresses.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_particle_addresses.restype = ct.c_bool

st_TrackJobNew_fetch_particle_addresses = \
    sixtracklib.st_TrackJobNew_fetch_particle_addresses
st_TrackJobNew_fetch_particle_addresses.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_fetch_particle_addresses.restype = st_arch_status_t

st_TrackJobNew_clear_particle_addresses = \
    sixtracklib.st_TrackJobNew_clear_particle_addresses
st_TrackJobNew_clear_particle_addresses.argtypes = [st_TrackJobBaseNew_p,
                                                    st_buffer_size_t]
st_TrackJobNew_clear_particle_addresses.restype = st_arch_status_t

st_TrackJobNew_clear_all_particle_addresses = \
    sixtracklib.st_TrackJobNew_clear_all_particle_addresses
st_TrackJobNew_clear_all_particle_addresses.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_clear_all_particle_addresses.restype = st_arch_status_t

st_TrackJobNew_get_particle_addresses = \
    sixtracklib.st_TrackJobNew_get_particle_addresses
st_TrackJobNew_get_particle_addresses.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_get_particle_addresses.restype = st_ParticlesAddr_p

st_TrackJobNew_get_ptr_particle_addresses_buffer = \
    sixtracklib.st_TrackJobNew_get_ptr_particle_addresses_buffer
st_TrackJobNew_get_ptr_particle_addresses_buffer.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_ptr_particle_addresses_buffer.restype = st_Buffer_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_is_in_debug_mode = sixtracklib.st_TrackJobNew_is_in_debug_mode
st_TrackJobNew_is_in_debug_mode.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_is_in_debug_mode.restype = ct.c_bool

st_TrackJobNew_enable_debug_mode = sixtracklib.st_TrackJobNew_enable_debug_mode
st_TrackJobNew_enable_debug_mode.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_enable_debug_mode.restype = st_arch_status_t

st_TrackJobNew_disable_debug_mode = \
    sixtracklib.st_TrackJobNew_disable_debug_mode
st_TrackJobNew_disable_debug_mode.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_disable_debug_mode.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_clear = sixtracklib.st_TrackJobNew_clear
st_TrackJobNew_clear.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_clear.restype = None

st_TrackJobNew_reset = sixtracklib.st_TrackJobNew_reset
st_TrackJobNew_reset.argtypes = [
    st_TrackJobBaseNew_p, st_Buffer_p, st_Buffer_p, st_Buffer_p]
st_TrackJobNew_reset.restype = st_arch_status_t

st_TrackJobNew_reset_particle_set = sixtracklib.st_TrackJobNew_reset
st_TrackJobNew_reset_particle_set.argtypes = [
    st_TrackJobBaseNew_p, st_Buffer_p, st_buffer_size_t, st_buffer_size_p,
    st_Buffer_p, st_Buffer_p]
st_TrackJobNew_reset_particle_set.restype = st_arch_status_t

st_TrackJobNew_reset_with_output = sixtracklib.st_TrackJobNew_reset_with_output
st_TrackJobNew_reset_with_output.argtypes = [
    st_TrackJobBaseNew_p,
    st_Buffer_p,
    st_Buffer_p,
    st_Buffer_p,
    st_buffer_size_t]
st_TrackJobNew_reset_with_output.restype = st_arch_status_t

st_TrackJobNew_reset_detailed = sixtracklib.st_TrackJobNew_reset_detailed
st_TrackJobNew_reset_detailed.argtypes = [
    st_TrackJobBaseNew_p,
    st_Buffer_p,
    st_buffer_size_t,
    st_buffer_size_p,
    st_Buffer_p,
    st_Buffer_p,
    st_buffer_size_t]
st_TrackJobNew_reset_detailed.restype = st_arch_status_t

st_TrackJobNew_select_particle_set = \
    sixtracklib.st_TrackJobNew_select_particle_set
st_TrackJobNew_select_particle_set.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_select_particle_set.restype = st_arch_status_t

st_TrackJobNew_assign_output_buffer = \
    sixtracklib.st_TrackJobNew_assign_output_buffer
st_TrackJobNew_assign_output_buffer.argtypes = [st_TrackJobBaseNew_p,
                                                st_Buffer_p]
st_TrackJobNew_assign_output_buffer.restype = st_arch_status_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_get_arch_id = sixtracklib.st_TrackJobNew_get_arch_id
st_TrackJobNew_get_arch_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_arch_id.restype = st_arch_id_t

st_TrackJobNew_has_arch_string = sixtracklib.st_TrackJobNew_has_arch_string
st_TrackJobNew_has_arch_string.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_arch_string.restype = ct.c_bool

st_TrackJobNew_get_arch_string = sixtracklib.st_TrackJobNew_get_arch_string
st_TrackJobNew_get_arch_string.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_arch_string.restype = ct.c_char_p

st_TrackJobNew_has_config_str = sixtracklib.st_TrackJobNew_has_config_str
st_TrackJobNew_has_config_str.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_config_str.restype = ct.c_bool

st_TrackJobNew_get_config_str = sixtracklib.st_TrackJobNew_get_config_str
st_TrackJobNew_get_config_str.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_config_str.restype = ct.c_char_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_get_num_particle_sets = \
    sixtracklib.st_TrackJobNew_get_num_particle_sets
st_TrackJobNew_get_num_particle_sets.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_num_particle_sets.restype = st_buffer_size_t

st_TrackJobNew_get_particle_set_indices_begin = \
    sixtracklib.st_TrackJobNew_get_particle_set_indices_begin
st_TrackJobNew_get_particle_set_indices_begin.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_particle_set_indices_begin.restype = st_buffer_size_p

st_TrackJobNew_get_particle_set_indices_end = \
    sixtracklib.st_TrackJobNew_get_particle_set_indices_end
st_TrackJobNew_get_particle_set_indices_end.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_particle_set_indices_end.restype = st_buffer_size_p

st_TrackJobNew_get_particle_set_index = \
    sixtracklib.st_TrackJobNew_get_particle_set_index
st_TrackJobNew_get_particle_set_index.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_get_particle_set_index.restype = st_buffer_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_get_total_num_of_particles = \
    sixtracklib.st_TrackJobNew_get_total_num_of_particles
st_TrackJobNew_get_total_num_of_particles.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_total_num_of_particles.restype = st_buffer_size_t

st_TrackJobNew_get_min_particle_id = \
    sixtracklib.st_TrackJobNew_get_min_particle_id
st_TrackJobNew_get_min_particle_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_min_particle_id.restype = st_particle_index_t

st_TrackJobNew_get_max_particle_id = \
    sixtracklib.st_TrackJobNew_get_max_particle_id
st_TrackJobNew_get_max_particle_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_max_particle_id.restype = st_particle_index_t

st_TrackJobNew_get_min_element_id = \
    sixtracklib.st_TrackJobNew_get_min_element_id
st_TrackJobNew_get_min_element_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_min_element_id.restype = st_particle_index_t

st_TrackJobNew_get_max_element_id = \
    sixtracklib.st_TrackJobNew_get_max_element_id
st_TrackJobNew_get_max_element_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_max_element_id.restype = st_particle_index_t

st_TrackJobNew_get_min_initial_turn_id = \
    sixtracklib.st_TrackJobNew_get_min_initial_turn_id
st_TrackJobNew_get_min_initial_turn_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_min_initial_turn_id.restype = st_particle_index_t

st_TrackJobNew_get_max_initial_turn_id = \
    sixtracklib.st_TrackJobNew_get_max_initial_turn_id
st_TrackJobNew_get_max_initial_turn_id.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_max_initial_turn_id.restype = st_particle_index_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_get_particles_buffer = \
    sixtracklib.st_TrackJobNew_get_particles_buffer
st_TrackJobNew_get_particles_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_particles_buffer.restype = st_Buffer_p

st_TrackJobNew_get_const_particles_buffer = \
    sixtracklib.st_TrackJobNew_get_const_particles_buffer
st_TrackJobNew_get_const_particles_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_const_particles_buffer.restype = st_Buffer_p

st_TrackJobNew_get_beam_elements_buffer = \
    sixtracklib.st_TrackJobNew_get_beam_elements_buffer
st_TrackJobNew_get_beam_elements_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_beam_elements_buffer.restype = st_Buffer_p

st_TrackJobNew_get_const_beam_elements_buffer = \
    sixtracklib.st_TrackJobNew_get_const_beam_elements_buffer
st_TrackJobNew_get_const_beam_elements_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_const_beam_elements_buffer.restype = st_Buffer_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_has_output_buffer = sixtracklib.st_TrackJobNew_has_output_buffer
st_TrackJobNew_has_output_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_output_buffer.restype = ct.c_bool

st_TrackJobNew_owns_output_buffer = \
    sixtracklib.st_TrackJobNew_owns_output_buffer
st_TrackJobNew_owns_output_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_owns_output_buffer.restype = ct.c_bool

st_TrackJobNew_has_elem_by_elem_output = \
    sixtracklib.st_TrackJobNew_has_elem_by_elem_output
st_TrackJobNew_has_elem_by_elem_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_elem_by_elem_output.restype = ct.c_bool

st_TrackJobNew_has_beam_monitor_output = \
    sixtracklib.st_TrackJobNew_has_beam_monitor_output
st_TrackJobNew_has_beam_monitor_output.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_beam_monitor_output.restype = ct.c_bool

st_TrackJobNew_get_beam_monitor_output_buffer_offset = \
    sixtracklib.st_TrackJobNew_get_beam_monitor_output_buffer_offset
st_TrackJobNew_get_beam_monitor_output_buffer_offset.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_beam_monitor_output_buffer_offset.restype = st_buffer_size_t

st_TrackJobNew_get_elem_by_elem_output_buffer_offset = \
    sixtracklib.st_TrackJobNew_get_elem_by_elem_output_buffer_offset
st_TrackJobNew_get_elem_by_elem_output_buffer_offset.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_elem_by_elem_output_buffer_offset.restype = st_buffer_size_t

st_TrackJobNew_get_num_elem_by_elem_turns = \
    sixtracklib.st_TrackJobNew_get_num_elem_by_elem_turns
st_TrackJobNew_get_num_elem_by_elem_turns.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_num_elem_by_elem_turns.restype = st_buffer_size_t

st_TrackJobNew_get_output_buffer = \
    sixtracklib.st_TrackJobNew_get_output_buffer
st_TrackJobNew_get_output_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_output_buffer.restype = st_Buffer_p

st_TrackJobNew_get_const_output_buffer = \
    sixtracklib.st_TrackJobNew_get_const_output_buffer
st_TrackJobNew_get_const_output_buffer.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_const_output_buffer.restype = st_Buffer_p

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_has_beam_monitors = \
    sixtracklib.st_TrackJobNew_has_beam_monitors
st_TrackJobNew_has_beam_monitors.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_beam_monitors.restype = ct.c_bool

st_TrackJobNew_get_num_beam_monitors = \
    sixtracklib.st_TrackJobNew_get_num_beam_monitors
st_TrackJobNew_get_num_beam_monitors.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_num_beam_monitors.restype = st_buffer_size_t

st_TrackJobNew_get_beam_monitor_indices_begin = \
    sixtracklib.st_TrackJobNew_get_beam_monitor_indices_begin
st_TrackJobNew_get_beam_monitor_indices_begin.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_beam_monitor_indices_begin.restype = st_buffer_size_p

st_TrackJobNew_get_beam_monitor_indices_end = \
    sixtracklib.st_TrackJobNew_get_beam_monitor_indices_end
st_TrackJobNew_get_beam_monitor_indices_end.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_beam_monitor_indices_end.restype = st_buffer_size_p

st_TrackJobNew_get_beam_monitor_index = \
    sixtracklib.st_TrackJobNew_get_beam_monitor_index
st_TrackJobNew_get_beam_monitor_index.argtypes = [
    st_TrackJobBaseNew_p, st_buffer_size_t]
st_TrackJobNew_get_beam_monitor_index.restype = st_buffer_size_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_has_elem_by_elem_config = \
    sixtracklib.st_TrackJobNew_has_elem_by_elem_config
st_TrackJobNew_has_elem_by_elem_config.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_has_elem_by_elem_config.restype = ct.c_bool

st_TrackJobNew_get_elem_by_elem_config = \
    sixtracklib.st_TrackJobNew_get_elem_by_elem_config
st_TrackJobNew_get_elem_by_elem_config.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_get_elem_by_elem_config.restype = st_ElemByElemConfig_p

st_TrackJobNew_is_elem_by_elem_config_rolling = \
    sixtracklib.st_TrackJobNew_is_elem_by_elem_config_rolling
st_TrackJobNew_is_elem_by_elem_config_rolling.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_is_elem_by_elem_config_rolling.restype = ct.c_bool

st_TrackJobNew_get_default_elem_by_elem_config_rolling_flag = \
    sixtracklib.st_TrackJobNew_get_default_elem_by_elem_config_rolling_flag
st_TrackJobNew_get_default_elem_by_elem_config_rolling_flag.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_default_elem_by_elem_config_rolling_flag.restype = ct.c_bool

st_TrackJobNew_set_default_elem_by_elem_config_rolling_flag = \
    sixtracklib.st_TrackJobNew_set_default_elem_by_elem_config_rolling_flag
st_TrackJobNew_set_default_elem_by_elem_config_rolling_flag.argtypes = [
    st_TrackJobBaseNew_p, ct.c_bool]
st_TrackJobNew_set_default_elem_by_elem_config_rolling_flag.restype = None

st_TrackJobNew_get_elem_by_elem_config_order = \
    sixtracklib.st_TrackJobNew_get_elem_by_elem_config_order
st_TrackJobNew_get_elem_by_elem_config_order.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_elem_by_elem_config_order.restype = \
    st_elem_by_elem_order_int_t

st_TrackJobNew_get_default_elem_by_elem_config_order = \
    sixtracklib.st_TrackJobNew_get_default_elem_by_elem_config_order
st_TrackJobNew_get_default_elem_by_elem_config_order.argtypes = [
    st_TrackJobBaseNew_p]
st_TrackJobNew_get_default_elem_by_elem_config_order.restype = \
    st_elem_by_elem_order_int_t

st_TrackJobNew_set_default_elem_by_elem_config_order = \
    sixtracklib.st_TrackJobNew_set_default_elem_by_elem_config_order
st_TrackJobNew_set_default_elem_by_elem_config_order.argtypes = [
    st_TrackJobBaseNew_p, st_elem_by_elem_order_int_t]
st_TrackJobNew_set_default_elem_by_elem_config_order.restype = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st_TrackJobNew_uses_controller = sixtracklib.st_TrackJobNew_uses_controller
st_TrackJobNew_uses_controller.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_uses_controller.restype = ct.c_bool

st_TrackJobNew_uses_arguments = sixtracklib.st_TrackJobNew_uses_arguments
st_TrackJobNew_uses_arguments.argtypes = [st_TrackJobBaseNew_p]
st_TrackJobNew_uses_arguments.restype = ct.c_bool

# ==============================================================================
# Cuda-Context methods

if SIXTRACKLIB_MODULES.get('cuda', False):
    # --------------------------------------------------------------------------
    # NS(CudaNodeInfo):

    st_CudaNodeInfo_p = ct.c_void_p
    st_NullCudaNodeInfo = ct.cast(0, st_CudaNodeInfo_p)

    st_cuda_dev_index_t = ct.c_int

    st_CudaNodeInfo_new = sixtracklib.st_CudaNodeInfo_new
    st_CudaNodeInfo_new.argtypes = [st_cuda_dev_index_t]
    st_CudaNodeInfo_new.restype = st_CudaNodeInfo_p

    st_CudaNodeInfo_new_detailed = sixtracklib.st_CudaNodeInfo_new_detailed
    st_CudaNodeInfo_new_detailed.argtypes = [
        st_cuda_dev_index_t,
        st_node_platform_id_t,
        st_node_device_id_t,
        st_node_index_t,
        ct.c_bool,
        ct.c_bool]
    st_CudaNodeInfo_new_detailed.restype = st_CudaNodeInfo_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaNodeInfo_get_cuda_device_index = \
        sixtracklib.st_CudaNodeInfo_get_cuda_device_index
    st_CudaNodeInfo_get_cuda_device_index.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_cuda_device_index.restype = st_cuda_dev_index_t

    st_CudaNodeInfo_get_pci_bus_id_str = \
        sixtracklib.st_CudaNodeInfo_get_pci_bus_id_str
    st_CudaNodeInfo_get_pci_bus_id_str.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_pci_bus_id_str.restype = ct.c_char_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaNodeInfo_get_warp_size = sixtracklib.st_CudaNodeInfo_get_warp_size
    st_CudaNodeInfo_get_warp_size.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_warp_size.restype = st_arch_size_t

    st_CudaNodeInfo_get_compute_capability = \
        sixtracklib.st_CudaNodeInfo_get_compute_capability
    st_CudaNodeInfo_get_compute_capability.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_compute_capability.restype = st_arch_size_t

    st_CudaNodeInfo_get_num_multiprocessors = \
        sixtracklib.st_CudaNodeInfo_get_num_multiprocessors
    st_CudaNodeInfo_get_num_multiprocessors.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_num_multiprocessors.restype = st_arch_size_t

    st_CudaNodeInfo_get_max_threads_per_block = \
        sixtracklib.st_CudaNodeInfo_get_max_threads_per_block
    st_CudaNodeInfo_get_max_threads_per_block.argtypes = [st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_max_threads_per_block.restype = st_arch_size_t

    st_CudaNodeInfo_get_max_threads_per_multiprocessor = \
        sixtracklib.st_CudaNodeInfo_get_max_threads_per_multiprocessor
    st_CudaNodeInfo_get_max_threads_per_multiprocessor.argtypes = [
        st_CudaNodeInfo_p]
    st_CudaNodeInfo_get_max_threads_per_multiprocessor.restype = st_arch_size_t

    # --------------------------------------------------------------------------
    # NS(CudaKernelConfig):

    st_CudaKernelConfig_p = ct.c_void_p
    st_NullCudaKernelConfig = ct.cast(0, st_CudaKernelConfig_p)

    # --------------------------------------------------------------------------
    # NS(CudaController):

    st_CudaArgBuffer_p = ct.c_void_p
    st_NullCudaArgBuffer = ct.cast(0, st_CudaArgBuffer_p)

    st_CudaArgument_p = ct.c_void_p
    st_NullCudaArgument = ct.cast(0, st_CudaArgument_p)

    st_CudaController_p = ct.c_void_p
    st_NullCudaController = ct.cast(0, st_CudaController_p)

    st_CudaController_create = sixtracklib.st_CudaController_create
    st_CudaController_create.argtypes = None
    st_CudaController_create.restype = st_CudaController_p

    st_CudaController_new = sixtracklib.st_CudaController_new
    st_CudaController_new.argtypes = [ct.c_char_p]
    st_CudaController_new.restype = st_CudaController_p

    st_CudaController_new_from_node_id = \
        sixtracklib.st_CudaController_new_from_node_id
    st_CudaController_new_from_node_id.argtypes = [st_NodeId_p]
    st_CudaController_new_from_node_id.restype = st_CudaController_p

    st_CudaController_new_from_node_index = \
        sixtracklib.st_CudaController_new_from_node_index
    st_CudaController_new_from_node_index.argtypes = [st_node_index_t]
    st_CudaController_new_from_node_index.restype = st_CudaController_p

    st_CudaController_new_from_platform_id_and_device_id = \
        sixtracklib.st_CudaController_new_from_platform_id_and_device_id
    st_CudaController_new_from_platform_id_and_device_id.argtypes = [
        st_node_platform_id_t, st_node_device_id_t]
    st_CudaController_new_from_platform_id_and_device_id.restype = \
        st_CudaController_p

    st_CudaController_new_from_cuda_device_index = \
        sixtracklib.st_CudaController_new_from_cuda_device_index
    st_CudaController_new_from_cuda_device_index.argtypes = [
        st_cuda_dev_index_t]
    st_CudaController_new_from_cuda_device_index.restype = st_CudaController_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_select_node_by_cuda_device_index = \
        sixtracklib.st_CudaController_select_node_by_cuda_device_index
    st_CudaController_select_node_by_cuda_device_index.argtypes = [
        st_CudaController_p, st_cuda_dev_index_t]
    st_CudaController_select_node_by_cuda_device_index.restype = \
        st_arch_status_t

    st_CudaController_select_node_by_cuda_pci_bus_id = \
        sixtracklib.st_CudaController_select_node_by_cuda_pci_bus_id
    st_CudaController_select_node_by_cuda_pci_bus_id.argtypes = [
        st_CudaController_p, ct.c_char_p]
    st_CudaController_select_node_by_cuda_pci_bus_id.restype = \
        st_arch_status_t

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_get_ptr_node_info_by_index = \
        sixtracklib.st_CudaController_get_ptr_node_info_by_index
    st_CudaController_get_ptr_node_info_by_index.argtypes = [
        st_CudaController_p, st_ctrl_size_t]
    st_CudaController_get_ptr_node_info_by_index.restype = st_CudaNodeInfo_p

    st_CudaController_get_ptr_node_info_by_platform_id_and_device_id = \
        sixtracklib.st_CudaController_get_ptr_node_info_by_platform_id_and_device_id
    st_CudaController_get_ptr_node_info_by_platform_id_and_device_id.argtypes = [
        st_CudaController_p, st_node_platform_id_t, st_node_device_id_t]
    st_CudaController_get_ptr_node_info_by_platform_id_and_device_id.restype = \
        st_CudaNodeInfo_p

    st_CudaController_get_ptr_node_info_by_node_id = \
        sixtracklib.st_CudaController_get_ptr_node_info_by_node_id
    st_CudaController_get_ptr_node_info_by_node_id.argtypes = [
        st_CudaController_p, st_NodeId_p]
    st_CudaController_get_ptr_node_info_by_node_id.restype = st_CudaNodeInfo_p

    st_CudaController_get_ptr_node_info = \
        sixtracklib.st_CudaController_get_ptr_node_info
    st_CudaController_get_ptr_node_info.argtypes = [
        st_CudaController_p, ct.c_char_p]
    st_CudaController_get_ptr_node_info.restype = st_CudaNodeInfo_p

    st_CudaController_get_ptr_node_info_by_cuda_dev_index = \
        sixtracklib.st_CudaController_get_ptr_node_info_by_cuda_dev_index
    st_CudaController_get_ptr_node_info_by_cuda_dev_index.argtypes = [
        st_CudaController_p, st_cuda_dev_index_t]
    st_CudaController_get_ptr_node_info_by_cuda_dev_index.restype = \
        st_CudaNodeInfo_p

    st_CudaController_get_ptr_node_info_by_pci_bus_id = \
        sixtracklib.st_CudaController_get_ptr_node_info_by_pci_bus_id
    st_CudaController_get_ptr_node_info_by_pci_bus_id.argtypes = [
        st_CudaController_p, ct.c_char_p]
    st_CudaController_get_ptr_node_info_by_pci_bus_id.restype = \
        st_CudaNodeInfo_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_add_kernel_config = \
        sixtracklib.st_CudaController_add_kernel_config
    st_CudaController_add_kernel_config.argtypes = [
        st_CudaController_p, st_CudaKernelConfig_p]
    st_CudaController_add_kernel_config.restype = st_kernel_id_t

    st_CudaController_add_kernel_config_detailed = \
        sixtracklib.st_CudaController_add_kernel_config_detailed
    st_CudaController_add_kernel_config_detailed.argtypes = [
        st_CudaController_p, ct.c_char_p, st_ctrl_size_t, st_ctrl_size_t,
        st_ctrl_size_t, st_ctrl_size_t, ct.c_char_p]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_get_ptr_kernel_config = \
        sixtracklib.st_CudaController_get_ptr_kernel_config
    st_CudaController_get_ptr_kernel_config.argtypes = [
        st_CudaController_p, st_kernel_id_t]
    st_CudaController_get_ptr_kernel_config.restype = st_CudaKernelConfig_p

    st_CudaController_get_ptr_kernel_config_by_kernel_name = \
        sixtracklib.st_CudaController_get_ptr_kernel_config_by_kernel_name
    st_CudaController_get_ptr_kernel_config_by_kernel_name.argtypes = [
        st_CudaController_p, ct.c_char_p]
    st_CudaController_get_ptr_kernel_config_by_kernel_name.restype = \
        st_CudaKernelConfig_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_is_managed_cobject_buffer_remapped = \
        sixtracklib.st_CudaController_is_managed_cobject_buffer_remapped
    st_CudaController_is_managed_cobject_buffer_remapped.argtypes = [
        st_CudaController_p, st_CudaArgBuffer_p, st_arch_size_t]
    st_CudaController_is_managed_cobject_buffer_remapped.restype = ct.c_bool

    st_CudaController_remap_managed_cobject_buffer = \
        sixtracklib.st_CudaController_remap_managed_cobject_buffer
    st_CudaController_remap_managed_cobject_buffer.argtypes = [
        st_CudaController_p, st_CudaArgBuffer_p, st_arch_size_t]
    st_CudaController_remap_managed_cobject_buffer.restype = st_arch_status_t

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaController_send_memory = sixtracklib.st_CudaController_send_memory
    st_CudaController_send_memory.argtypes = [
        st_CudaController_p,
        st_CudaArgument_p,
        ct.c_void_p,
        st_arch_size_t]
    st_CudaController_send_memory.restype = st_arch_status_t

    st_CudaController_receive_memory = \
        sixtracklib.st_CudaController_receive_memory
    st_CudaController_receive_memory.argtypes = [
        st_CudaController_p, ct.c_void_p, st_CudaArgument_p, st_arch_size_t]
    st_CudaController_receive_memory.restype = st_arch_status_t

    # --------------------------------------------------------------------------
    # NS(CudaArgument):

    st_arch_debugging_t = ct.c_uint64
    st_arch_debugging_p = ct.POINTER(st_arch_debugging_t)

    st_CudaArgument_new = sixtracklib.st_CudaArgument_new
    st_CudaArgument_new.argtypes = [st_CudaController_p]
    st_CudaArgument_new.restype = st_CudaArgument_p

    st_CudaArgument_new_from_buffer = \
        sixtracklib.st_CudaArgument_new_from_buffer
    st_CudaArgument_new_from_buffer.argtypes = [
        st_Buffer_p, st_CudaController_p]
    st_CudaArgument_new_from_buffer.restype = st_CudaArgument_p

    st_CudaArgument_new_from_raw_argument = \
        sixtracklib.st_CudaArgument_new_from_raw_argument
    st_CudaArgument_new_from_raw_argument.argtypes = [
        ct.c_void_p, st_ctrl_size_t, st_CudaController_p]
    st_CudaArgument_new_from_raw_argument.restype = st_CudaArgument_p

    st_CudaArgument_new_from_size = \
        sixtracklib.st_CudaArgument_new_from_size
    st_CudaArgument_new_from_size.argtypes = [
        st_ctrl_size_t, st_CudaController_p]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_Argument_has_cuda_arg_buffer = \
        sixtracklib.st_Argument_has_cuda_arg_buffer
    st_Argument_has_cuda_arg_buffer.argtypes = [st_ArgumentBase_p]
    st_Argument_has_cuda_arg_buffer.restype = ct.c_bool

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaArgument_get_cuda_arg_buffer = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer
    st_CudaArgument_get_cuda_arg_buffer.argtypes = [st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer.restype = st_CudaArgBuffer_p

    st_CudaArgument_get_const_cuda_arg_buffer = \
        sixtracklib.st_CudaArgument_get_const_cuda_arg_buffer
    st_CudaArgument_get_const_cuda_arg_buffer.argtypes = [st_CudaArgument_p]
    st_CudaArgument_get_const_cuda_arg_buffer.restype = st_CudaArgBuffer_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin
    st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin.argtypes = [
        st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin.restype = \
        st_uchar_p

    st_CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin
    st_CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin.argtypes \
        = [st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin.restype \
        = st_const_uchar_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin
    st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin.argtypes = [
        st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin.restype = \
        st_arch_debugging_p

    st_CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin
    st_CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin.argtypes \
        = [st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin.restype \
        = st_arch_debugging_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin
    st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin.argtypes = [
        st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin.restype = \
        st_ElemByElemConfig_p

    st_CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin
    st_CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin.argtypes = [
        st_CudaArgument_p]
    st_CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin.restype = \
        st_ElemByElemConfig_p

    # ==========================================================================
    # NS(CudaTrackJob):

    st_CudaTrackJob_p = ct.c_void_p
    st_NullCudaTrackJob = ct.cast(0, st_CudaTrackJob_p)

    st_CudaTrackJob_get_num_available_nodes = \
        sixtracklib.st_CudaTrackJob_get_num_available_nodes
    st_CudaTrackJob_get_num_available_nodes.argtypes = None
    st_CudaTrackJob_get_num_available_nodes.restype = st_ctrl_size_t

    _st_CudaTrackJob_get_available_node_ids_list = \
        sixtracklib.st_CudaTrackJob_get_available_node_ids_list
    _st_CudaTrackJob_get_available_node_ids_list.argtypes = [
        st_ctrl_size_t, st_NodeId_p]
    _st_CudaTrackJob_get_available_node_ids_list.restype = st_ctrl_size_t

    st_CudaTrackJob_get_available_node_indices_list = \
        sixtracklib.st_CudaTrackJob_get_available_node_indices_list
    st_CudaTrackJob_get_available_node_indices_list.argtypes = [
        st_ctrl_size_t, st_node_index_p]
    st_CudaTrackJob_get_available_node_indices_list.restype = st_ctrl_size_t

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_create = sixtracklib.st_CudaTrackJob_create
    st_CudaTrackJob_create.argtypes = None
    st_CudaTrackJob_create.restype = st_CudaTrackJob_p

    st_CudaTrackJob_new_from_config_str = \
        sixtracklib.st_CudaTrackJob_new_from_config_str
    st_CudaTrackJob_new_from_config_str.argtypes = [ct.c_char_p]
    st_CudaTrackJob_new_from_config_str.restype = st_CudaTrackJob_p

    st_CudaTrackJob_new = sixtracklib.st_CudaTrackJob_new
    st_CudaTrackJob_new.argtypes = [ct.c_char_p, st_Buffer_p, st_Buffer_p]
    st_CudaTrackJob_new.restype = st_CudaTrackJob_p

    st_CudaTrackJob_new_with_output = \
        sixtracklib.st_CudaTrackJob_new_with_output
    st_CudaTrackJob_new_with_output.argtypes = [
        ct.c_char_p, st_Buffer_p, st_Buffer_p, st_Buffer_p, st_buffer_size_t]
    st_CudaTrackJob_new_with_output.restype = st_CudaTrackJob_p

    st_CudaTrackJob_new_detailed = sixtracklib.st_CudaTrackJob_new_detailed
    st_CudaTrackJob_new_detailed.argtypes = [
        ct.c_char_p,
        st_Buffer_p,
        st_buffer_size_t,
        st_buffer_size_p,
        st_Buffer_p,
        st_Buffer_p,
        st_buffer_size_t,
        ct.c_char_p]
    st_CudaTrackJob_new_detailed.restype = st_CudaTrackJob_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_controller = sixtracklib.st_CudaTrackJob_has_controller
    st_CudaTrackJob_has_controller.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_controller.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_controller = \
        sixtracklib.st_CudaTrackJob_get_ptr_controller
    st_CudaTrackJob_get_ptr_controller.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_controller.restype = st_CudaController_p

    st_CudaTrackJob_get_ptr_const_controller = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_controller
    st_CudaTrackJob_get_ptr_const_controller.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_controller.restype = st_CudaController_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_particles_arg = \
        sixtracklib.st_CudaTrackJob_has_particles_arg
    st_CudaTrackJob_has_particles_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_particles_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_particles_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_particles_arg
    st_CudaTrackJob_get_ptr_particles_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_particles_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_particles_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_particles_arg
    st_CudaTrackJob_get_ptr_const_particles_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_particles_arg.restype = st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_beam_elements_arg = \
        sixtracklib.st_CudaTrackJob_has_beam_elements_arg
    st_CudaTrackJob_has_beam_elements_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_beam_elements_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_beam_elements_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_beam_elements_arg
    st_CudaTrackJob_get_ptr_beam_elements_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_beam_elements_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_beam_elements_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_beam_elements_arg
    st_CudaTrackJob_get_ptr_const_beam_elements_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_beam_elements_arg.restype = st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_output_arg = \
        sixtracklib.st_CudaTrackJob_has_output_arg
    st_CudaTrackJob_has_output_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_output_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_output_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_output_arg
    st_CudaTrackJob_get_ptr_output_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_output_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_output_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_output_arg
    st_CudaTrackJob_get_ptr_const_output_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_output_arg.restype = st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # st_CudaTrackJob_has_particles_arg = \
    # sixtracklib.st_CudaTrackJob_has_particles_arg
    #st_CudaTrackJob_has_particles_arg.argtypes = [ st_CudaTrackJob_p ]
    #st_CudaTrackJob_has_particles_arg.restype = ct.c_bool

    # st_CudaTrackJob_get_ptr_particles_arg = \
    # sixtracklib.st_CudaTrackJob_get_ptr_particles_arg
    #st_CudaTrackJob_get_ptr_particles_arg.argtypes = [ st_CudaTrackJob_p ]
    #st_CudaTrackJob_get_ptr_particles_arg.restype = st_CudaArgument_p

    # st_CudaTrackJob_get_ptr_const_particles_arg = \
    # sixtracklib.st_CudaTrackJob_get_ptr_const_particles_arg
    #st_CudaTrackJob_get_ptr_const_particles_arg.argtypes = [ st_CudaTrackJob_p ]
    #st_CudaTrackJob_get_ptr_const_particles_arg.restype = st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_elem_by_elem_config_arg = \
        sixtracklib.st_CudaTrackJob_has_elem_by_elem_config_arg
    st_CudaTrackJob_has_elem_by_elem_config_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_has_elem_by_elem_config_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_elem_by_elem_config_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_elem_by_elem_config_arg
    st_CudaTrackJob_get_ptr_elem_by_elem_config_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_elem_by_elem_config_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_elem_by_elem_config_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_elem_by_elem_config_arg
    st_CudaTrackJob_get_ptr_const_elem_by_elem_config_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_elem_by_elem_config_arg.restype = \
        st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_debug_register_arg = \
        sixtracklib.st_CudaTrackJob_has_debug_register_arg
    st_CudaTrackJob_has_debug_register_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_debug_register_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_debug_register_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_debug_register_arg
    st_CudaTrackJob_get_ptr_debug_register_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_debug_register_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_debug_register_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_debug_register_arg
    st_CudaTrackJob_get_ptr_const_debug_register_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_debug_register_arg.restype = \
        st_CudaArgument_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    st_CudaTrackJob_has_particles_addr_arg = \
        sixtracklib.st_CudaTrackJob_has_particles_addr_arg
    st_CudaTrackJob_has_particles_addr_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_has_particles_addr_arg.restype = ct.c_bool

    st_CudaTrackJob_get_ptr_particles_addr_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_particles_addr_arg
    st_CudaTrackJob_get_ptr_particles_addr_arg.argtypes = [st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_particles_addr_arg.restype = st_CudaArgument_p

    st_CudaTrackJob_get_ptr_const_particles_addr_arg = \
        sixtracklib.st_CudaTrackJob_get_ptr_const_particles_addr_arg
    st_CudaTrackJob_get_ptr_const_particles_addr_arg.argtypes = [
        st_CudaTrackJob_p]
    st_CudaTrackJob_get_ptr_const_particles_addr_arg.restype = \
        st_CudaArgument_p

# -----------------------------------------------------------------------------
# Cl-Context methods

if SIXTRACKLIB_MODULES.get('opencl', False):

    st_ClContext_create = sixtracklib.st_ClContext_create
    st_ClContext_create.restype = st_Context_p

    st_ClContextBase_select_node = sixtracklib.st_ClContextBase_select_node
    st_ClContextBase_select_node.argtypes = [st_Context_p, ct.c_char_p]
    st_ClContextBase_select_node.restype = None

    st_ClContextBase_print_nodes_info = \
        sixtracklib.st_ClContextBase_print_nodes_info
    st_ClContextBase_print_nodes_info.argtypes = [st_Context_p]
    st_ClContextBase_print_nodes_info.restype = None

    st_ClContextBase_delete = sixtracklib.st_ClContextBase_delete
    st_ClContextBase_delete.argtypes = [st_Context_p]
    st_ClContextBase_delete.restype = None

# ------------------------------------------------------------------------------
# Stand-alone tracking functions (CPU only)

st_Track_all_particles_until_turn = \
    sixtracklib.st_Track_all_particles_until_turn
st_Track_all_particles_until_turn.restype = st_track_status_t
st_Track_all_particles_until_turn.argtypes = [
    st_Particles_p, st_Buffer_p, st_particle_index_t]

st_Track_all_particles_element_by_element_until_turn = \
    sixtracklib.st_Track_all_particles_element_by_element_until_turn
st_Track_all_particles_element_by_element_until_turn.restype = st_track_status_t
st_Track_all_particles_element_by_element_until_turn.argtypes = [
    st_Particles_p, st_ElemByElemConfig_p, st_Buffer_p, ct.c_int64]
