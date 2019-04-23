import ctypes as ct
from . import config as stconf
from .particles import Particles as st_Particles
from .config import SIXTRACKLIB_MODULES
from cobjects import CBuffer

sixtracklib = ct.CDLL(stconf.SHARED_LIB)

# C-API Types

st_Null = ct.cast(0, ct.c_void_p)
st_NullChar = ct.cast(0, ct.c_char_p)

st_Context_p = ct.c_void_p
st_uint64_p = ct.POINTER(ct.c_uint64)
st_uchar_p = ct.POINTER(ct.c_ubyte)

st_double_p = ct.POINTER(ct.c_double)
st_int64_p = ct.POINTER(ct.c_int64)

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

st_Buffer_new_from_file = sixtracklib.st_Buffer_new_from_file
st_Buffer_new_from_file.argtypes = [ct.c_char_p]
st_Buffer_new_from_file.restype = st_Buffer_p

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

st_ParticlesAddr_p = ct.POINTER(st_ParticlesAddr)
st_NullParticlesAddr = ct.cast(0, st_ParticlesAddr_p)


def st_Particles_cbuffer_get_particles(cbuffer, obj_index):
    return ct.cast(cbuffer.get_object_address(obj_index), st_Particles_p)


st_Particles_preset = sixtracklib.st_Particles_preset_ext
st_Particles_preset.argtypes = [st_Particles_p]
st_Particles_preset.restype = st_Particles_p

st_Particles_get_num_of_particles = \
    sixtracklib.st_Particles_get_num_of_particles_ext
st_Particles_get_num_of_particles.argtpyes = [st_Particles_p]
st_Particles_get_num_of_particles.restype = ct.c_int64

st_Particles_copy_single = sixtracklib.st_Particles_copy_single_ext
st_Particles_copy_single.restype = ct.c_bool
st_Particles_copy_single.argptypes = [
    st_Particles_p, ct.c_int64, st_Particles_p, ct.c_int64]

st_Particles_copy_range = sixtracklib.st_Particles_copy_range_ext
st_Particles_copy_range.restype = ct.c_bool
st_Particles_copy_range.argtypes = [
    st_Particles_p, st_Particles_p, ct.c_int64, ct.c_int64, ct.c_int64]

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
st_Particles_buffer_get_total_num_of_particles.restype = ct.c_int64
st_Particles_buffer_get_total_num_of_particles.argtypes = [st_Buffer_p]

st_Particles_buffer_get_num_of_particle_blocks = \
    sixtracklib.st_Particles_buffer_get_num_of_particle_blocks_ext
st_Particles_buffer_get_num_of_particle_blocks.restype = ct.c_uint64
st_Particles_buffer_get_num_of_particle_blocks.argtypes = [st_Buffer_p]

st_Particles_buffer_get_particles = \
    sixtracklib.st_Particles_buffer_get_particles_ext
st_Particles_buffer_get_particles.restype = st_Particles_p
st_Particles_buffer_get_particles.argtypes = [st_Buffer_p, ct.c_uint64]

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
st_Particles_get_required_num_slots.restype = ct.c_uint64
st_Particles_get_required_num_slots.argtypes = [st_Buffer_p, ct.c_uint64]

st_Particles_get_required_num_dataptrs = \
    sixtracklib.st_Particles_get_required_num_dataptrs_ext
st_Particles_get_required_num_dataptrs.restype = ct.c_uint64
st_Particles_get_required_num_dataptrs.argtypes = [st_Buffer_p, ct.c_uint64]

st_Particles_can_be_added = sixtracklib.st_Particles_can_be_added_ext
st_Particles_can_be_added.restype = ct.c_bool
st_Particles_can_be_added.argtypes = [
    st_Buffer_p, ct.c_uint64, st_uint64_p, st_uint64_p, st_uint64_p]

st_Particles_new = sixtracklib.st_Particles_new_ext
st_Particles_new.argtypes = [st_Buffer_p, ct.c_uint64]
st_Particles_new.restype = st_Particles_p

st_Particles_add = sixtracklib.st_Particles_add_ext
st_Particles_add.restype = st_Particles_p
st_Particles_add.argtypes = [
    st_Buffer_p,
    ct.c_uint64,
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

st_ParticlesAddr_preset = sixtracklib.st_ParticlesAddr_preset
st_ParticlesAddr_preset.argtypes = [st_ParticlesAddr_p]
st_ParticlesAddr_preset.restype = st_ParticlesAddr_p

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
st_OutputBuffer_required_for_tracking.argptypes = [
    st_Particles_p,
    st_Buffer_p,
    ct.c_uint64]

st_OutputBuffer_required_for_tracking_of_particle_sets = \
    sixtracklib.st_OutputBuffer_required_for_tracking_of_particle_sets
st_OutputBuffer_required_for_tracking_of_particle_sets.restype = ct.c_int32
st_OutputBuffer_required_for_tracking_of_particle_sets.argptypes = [
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

# -----------------------------------------------------------------------------
# Cuda-Context methods

if SIXTRACKLIB_MODULES.get('cuda', False):
    st_CudaContext_p = ct.c_void_p
    st_NullCudaContext = ct.cast(0, st_CudaContext_p)

    st_CudaArgument_p = ct.c_void_p
    st_NullCudaArgument = ct.cast(0, st_CudaArgument_p)

    st_CudaContext_create = sixtracklib.st_CudaContext_create
    st_CudaContext_create.argtypes = None
    st_CudaContext_create.restype = st_CudaContext_p

    st_CudaContext_delete = sixtracklib.st_CudaContext_delete
    st_CudaContext_delete.argtypes = [st_CudaContext_p]
    st_CudaContext_delete.restype = None

    st_CudaArgument_new = sixtracklib.st_CudaArgument_new
    st_CudaArgument_new.argtypes = [st_CudaContext_p]
    st_CudaArgument_new.restype = st_CudaArgument_p

    st_CudaArgument_new_from_buffer = \
        sixtracklib.st_CudaArgument_new_from_buffer
    st_CudaArgument_new_from_buffer.restype = st_CudaArgument_p
    st_CudaArgument_new_from_buffer.argtypes = [st_Buffer_p, st_CudaContext_p]

    st_CudaArgument_new_from_size = sixtracklib.st_CudaArgument_new_from_size
    st_CudaArgument_new_from_size.restype = st_CudaArgument_p
    st_CudaArgument_new_from_size.argtypes = [ct.c_uint64, st_CudaContext_p]

    st_CudaArgument_new_from_memory = \
        sixtracklib.st_CudaArgument_new_from_memory
    st_CudaArgument_new_from_memory.restype = st_CudaArgument_p
    st_CudaArgument_new_from_memory.argtypes = [
        ct.c_void_p, ct.c_uint64, st_CudaContext_p]

    st_CudaArgument_delete = sixtracklib.st_CudaArgument_delete
    st_CudaArgument_delete.restype = None
    st_CudaArgument_delete.argtypes = [st_CudaArgument_p]

    st_CudaArgument_send_buffer = sixtracklib.st_CudaArgument_send_buffer
    st_CudaArgument_send_buffer.restype = ct.c_bool
    st_CudaArgument_send_buffer.argtypes = [st_CudaArgument_p, st_Buffer_p]

    st_CudaArgument_send_memory = sixtracklib.st_CudaArgument_send_memory
    st_CudaArgument_send_memory.restype = ct.c_bool
    st_CudaArgument_send_memory.argtypes = [
        st_CudaArgument_p, ct.c_void_p, ct.c_uint64]

    st_CudaArgument_receive_buffer = sixtracklib.st_CudaArgument_receive_buffer
    st_CudaArgument_receive_buffer.restype = ct.c_bool
    st_CudaArgument_receive_buffer.argtypes = [st_CudaArgument_p, st_Buffer_p]

    st_CudaArgument_receive_memory = sixtracklib.st_CudaArgument_receive_memory
    st_CudaArgument_receive_memory.restype = ct.c_bool
    st_CudaArgument_receive_memory.argtypes = [
        st_CudaArgument_p, ct.c_void_p, ct.c_uint64]

    st_CudaArgument_get_arg_buffer = \
        sixtracklib.st_CudaArgument_get_cuda_arg_buffer
    st_CudaArgument_get_arg_buffer.argtypes = [st_CudaArgument_p]
    st_CudaArgument_get_arg_buffer.restype = ct.c_void_p

    st_CudaArgument_uses_cobjects_buffer = \
        sixtracklib.st_CudaArgument_uses_cobjects_buffer
    st_CudaArgument_uses_cobjects_buffer.restype = ct.c_bool
    st_CudaArgument_uses_cobjects_buffer.argtypes = [st_CudaArgument_p]

    st_CudaArgument_get_cobjects_buffer =  \
        sixtracklib.st_CudaArgument_get_cobjects_buffer
    st_CudaArgument_get_cobjects_buffer.restype = st_Buffer_p
    st_CudaArgument_get_cobjects_buffer.argtypes = [st_CudaArgument_p]

    st_CudaArgument_uses_raw_argument = \
        sixtracklib.st_CudaArgument_uses_raw_argument
    st_CudaArgument_uses_raw_argument.restype = ct.c_bool
    st_CudaArgument_uses_raw_argument.argtypes = [st_CudaArgument_p]

    st_CudaArgument_get_ptr_raw_argument = \
        sixtracklib.st_CudaArgument_get_ptr_raw_argument
    st_CudaArgument_get_ptr_raw_argument.restype = ct.c_void_p
    st_CudaArgument_get_ptr_raw_argument.argtypes = [st_CudaArgument_p]

    st_CudaArgument_get_size = sixtracklib.st_CudaArgument_get_size
    st_CudaArgument_get_size.restype = ct.c_uint64
    st_CudaArgument_get_size.argtypes = [st_CudaArgument_p]

    st_CudaArgument_get_capacity = sixtracklib.st_CudaArgument_get_capacity
    st_CudaArgument_get_capacity.restype = ct.c_uint64
    st_CudaArgument_get_capacity.argtypes = [st_CudaArgument_p]

    st_CudaArgument_has_argument_buffer = \
        sixtracklib.st_CudaArgument_has_argument_buffer
    st_CudaArgument_has_argument_buffer.restype = ct.c_bool
    st_CudaArgument_has_argument_buffer.argtypes = [st_CudaArgument_p]

    st_CudaArgument_requires_argument_buffer = \
        sixtracklib.st_CudaArgument_requires_argument_buffer
    st_CudaArgument_requires_argument_buffer.restype = ct.c_bool
    st_CudaArgument_requires_argument_buffer.argtypes = [st_CudaArgument_p]

    st_CudaArgument_get_type_id = sixtracklib.st_CudaArgument_get_type_id
    st_CudaArgument_get_type_id.restype = ct.c_uint64
    st_CudaArgument_get_type_id.argtypes = [st_CudaArgument_p]

    # Extract particles API for CUDA

    st_Particles_extract_addresses_cuda = \
        sixtracklib.st_Particles_extract_addresses_cuda
    st_Particles_extract_addresses_cuda.argtypes = [ct.c_void_p, ct.c_void_p]
    st_Particles_extract_addresses_cuda.restype = ct.c_int32

    # Stand-alone tracking functions for CUDA

    st_Track_particles_line_cuda_on_grid = \
        sixtracklib.st_Track_particles_line_cuda_on_grid
    st_Track_particles_line_cuda_on_grid.restype = ct.c_int32
    st_Track_particles_line_cuda_on_grid.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint64,
        ct.c_uint64,
        ct.c_bool,
        ct.c_uint64,
        ct.c_uint64]

    st_Track_particles_line_cuda = sixtracklib.st_Track_particles_line_cuda
    st_Track_particles_line_cuda.restype = ct.c_int32
    st_Track_particles_line_cuda.argtypes = [ct.c_void_p, ct.c_void_p,
                                             ct.c_uint64, ct.c_uint64, ct.c_bool]

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
st_Track_all_particles_until_turn.restype = ct.c_int32
st_Track_all_particles_until_turn.argtypes = [
    st_Particles_p, st_Buffer_p, ct.c_int64]

st_Track_all_particles_element_by_element_until_turn = \
    sixtracklib.st_Track_all_particles_element_by_element_until_turn
st_Track_all_particles_element_by_element_until_turn.restype = ct.c_int32
st_Track_all_particles_element_by_element_until_turn.argtypes = [
    st_Particles_p, st_Buffer_p, ct.c_int64, st_Particles_p]

