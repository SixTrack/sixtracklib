import ctypes as ct
from . import sixtracklibconf as stconf
from .particles import Particles as st_Particles

sixtracklib = ct.CDLL( stconf.SHARED_LIB )

# C-API Types

st_Null=ct.cast(0,ct.c_void_p)
st_NullChar=ct.cast(0,ct.c_char_p)

st_Context_p  = ct.c_void_p
st_TrackJob_p = ct.c_void_p
st_uint64_p   = ct.POINTER( ct.c_uint64 )
st_uchar_p    = ct.POINTER( ct.c_ubyte )

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
st_NullBuffer=ct.cast(0, st_Buffer_p)

st_Buffer_preset = sixtracklib.st_Buffer_preset_ext
st_Buffer_preset.argtypes = [ st_Buffer_p ]
st_Buffer_preset.restype  = st_Buffer_p

st_Buffer_new = sixtracklib.st_Buffer_new
st_Buffer_new.argtypes = [ ct.c_uint64 ]
st_Buffer_new.restype  = st_Buffer_p

st_Buffer_new_on_memory = sixtracklib.st_Buffer_new_on_memory
st_Buffer_new_on_memory.argtypes = [ st_uchar_p, ct.c_uint64]
st_Buffer_new_on_memory.restype = st_Buffer_p

st_Buffer_new_on_data = sixtracklib.st_Buffer_new_on_data
st_Buffer_new_on_data.argtypes = [ st_uchar_p, ct.c_uint64 ]
st_Buffer_new_on_data.restype  = st_Buffer_p

st_Buffer_new_from_file = sixtracklib.st_Buffer_new_from_file
st_Buffer_new_from_file.argtypes = [ ct.c_char_p ]
st_Buffer_new_from_file.restype  = st_Buffer_p

st_Buffer_init_from_data = sixtracklib.st_Buffer_init_from_data_ext
st_Buffer_init_from_data.argtypes = [ st_Buffer_p, st_uchar_p, ct.c_uint64 ]
st_Buffer_init_from_data.restypes = ct.c_int32

st_Buffer_get_slot_size = sixtracklib.st_Buffer_get_slot_size_ext
st_Buffer_get_slot_size.argtypes = [ st_Buffer_p, ]
st_Buffer_get_slot_size.restype  = ct.c_uint64

st_Buffer_get_num_of_objects = sixtracklib.st_Buffer_get_num_of_objects_ext
st_Buffer_get_num_of_objects.argtypes = [ st_Buffer_p, ]
st_Buffer_get_num_of_objects.restype  = ct.c_uint64

st_Buffer_get_size = sixtracklib.st_Buffer_get_size_ext
st_Buffer_get_size.argtypes = [ st_Buffer_p ]
st_Buffer_get_size.restype  = ct.c_uint64

st_Buffer_get_capacity = sixtracklib.st_Buffer_get_capacity_ext
st_Buffer_get_capacity.argtypes = [ st_Buffer_p ]
st_Buffer_get_capacity.restype  = ct.c_uint64

st_Buffer_free   =  sixtracklib.st_Buffer_free_ext
st_Buffer_free.argtypes = [ st_Buffer_p ]
st_Buffer_free.restype  = None

st_Buffer_delete =  sixtracklib.st_Buffer_delete
st_Buffer_delete.argtypes = [st_Buffer_p]
st_Buffer_delete.restype  = None

# Helper Classes

class BufferAdapter(object):
    def __init__(self, cbuffer):
        self.cbuffer  = cbuffer
        self.stbuffer = st_Buffer;

        buff = ct.cast(self.cbuffer.base, ct.c_char_p)
        nn = self.cbuffer.size
        ret = st_Buffer_init_from_data( ct.POINTER( self.stbuffer ), buff, nn )

    def info(self):
        print(self.stbuffer.contents)

    def __del__(self):
        print(f"De-allocate STBuffer {self.stbuffer}")
        st_Buffer_delete(self.stbuffer)

# -----------------------------------------------------------------------------
# BeamMonitor objects

st_BeamMonitor_insert_end_of_turn_monitors = \
    sixtracklib.st_BeamMonitor_insert_end_of_turn_monitors_at_pos

st_BeamMonitor_insert_end_of_turn_monitors.argtypes = [ st_Buffer_p,
    ct.c_int64, ct.c_int64, ct.c_int64, ct.c_int64, ct.c_uint64, ]

st_BeamMonitor_insert_end_of_turn_monitors.restype = ct.c_int32

# -----------------------------------------------------------------------------
# OutputBuffer bindings

st_OutputBuffer_calculate_output_buffer_params = \
    sixtracklib.st_OutputBuffer_calculate_output_buffer_params_for_particles_sets

st_OutputBuffer_calculate_output_buffer_params.argtypes = [ st_Buffer_p,
    st_Buffer_p, ct.c_uint64, st_uint64_p, ct.c_uint64, st_uint64_p,
    st_uint64_p, st_uint64_p, st_uint64_p, ct.c_uint64 ]

st_OutputBuffer_calculate_output_buffer_params.restype  = ct.c_int32

# -----------------------------------------------------------------------------
# TrackJob objects

st_TrackJob_create = sixtracklib.st_TrackJobCpu_create
st_TrackJob_create.argtypes = [ ct.c_char_p, ct.c_char_p ]
st_TrackJob_create.restype  = st_TrackJob_p


st_TrackJob_new = sixtracklib.st_TrackJob_new
st_TrackJob_new.argtypes = [ ct.c_char_p, st_Buffer_p, st_Buffer_p, ct.c_char_p ]
st_TrackJob_new.restype  = st_TrackJob_p


st_TrackJob_new_with_output = sixtracklib.st_TrackJob_new_with_output
st_TrackJob_new_with_output.argtypes = [
    ct.c_char_p, st_Buffer_p, st_Buffer_p, st_Buffer_p,
    ct.c_uint64, ct.c_char_p ]
st_TrackJob_new_with_output.restype = st_TrackJob_p


st_TrackJob_delete = sixtracklib.st_TrackJob_delete
st_TrackJob_delete.argtypes = [ st_TrackJob_p ]
st_TrackJob_delete.restype  = None


st_TrackJob_track_until  = sixtracklib.st_TrackJob_track_until
st_TrackJob_track_until.argtypes  = [ st_TrackJob_p, ct.c_uint64 ]
st_TrackJob_track_until.restype   = ct.c_int32


st_TrackJob_track_elem_by_elem  = sixtracklib.st_TrackJob_track_elem_by_elem
st_TrackJob_track_elem_by_elem.argtypes  = [ st_TrackJob_p, ct.c_uint64 ]
st_TrackJob_track_elem_by_elem.restype   = ct.c_int32


st_TrackJob_collect = sixtracklib.st_TrackJob_collect
st_TrackJob_collect.argtypes = [ st_TrackJob_p ]
st_TrackJob_collect.restype  = None


st_TrackJob_get_type_id = sixtracklib.st_TrackJob_get_type_id
st_TrackJob_get_type_id.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_type_id.restype  = ct.c_int64


st_TrackJob_get_type_str = sixtracklib.st_TrackJob_get_type_str
st_TrackJob_get_type_str.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_type_str.restype  = ct.c_char_p


st_TrackJob_has_output_buffer = sixtracklib.st_TrackJob_has_output_buffer
st_TrackJob_has_output_buffer.argtypes = [ st_TrackJob_p ]
st_TrackJob_has_output_buffer.restype  = ct.c_bool


st_TrackJob_owns_output_buffer = sixtracklib.st_TrackJob_owns_output_buffer
st_TrackJob_owns_output_buffer.argtypes = [ st_TrackJob_p ]
st_TrackJob_owns_output_buffer.restype  = ct.c_bool


st_TrackJob_get_output_buffer = sixtracklib.st_TrackJob_get_output_buffer
st_TrackJob_get_output_buffer.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_output_buffer.restype  = st_Buffer_p


st_TrackJob_has_elem_by_elem_output = \
    sixtracklib.st_TrackJob_has_elem_by_elem_output
st_TrackJob_has_elem_by_elem_output.argtypes = [ st_TrackJob_p ]
st_TrackJob_has_elem_by_elem_output.restype  = ct.c_bool


st_TrackJob_get_elem_by_elem_output_buffer_offset = \
    sixtracklib.st_TrackJob_get_elem_by_elem_output_buffer_offset
st_TrackJob_get_elem_by_elem_output_buffer_offset.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_elem_by_elem_output_buffer_offset.restype  = ct.c_uint64


st_TrackJob_has_beam_monitor_output = \
    sixtracklib.st_TrackJob_has_beam_monitor_output
st_TrackJob_has_beam_monitor_output.argtypes = [ st_TrackJob_p ]
st_TrackJob_has_beam_monitor_output.restype  = ct.c_bool


st_TrackJob_get_num_beam_monitors = \
    sixtracklib.st_TrackJob_get_num_beam_monitors
st_TrackJob_get_num_beam_monitors.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_num_beam_monitors.restype  = ct.c_uint64


st_TrackJob_get_beam_monitor_output_buffer_offset = \
    sixtracklib.st_TrackJob_get_beam_monitor_output_buffer_offset
st_TrackJob_get_beam_monitor_output_buffer_offset.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_beam_monitor_output_buffer_offset.restype  = ct.c_uint64


# -----------------------------------------------------------------------------

st_ClContext_create = sixtracklib.st_ClContext_create
st_ClContext_create.restype = st_Context_p

st_ClContextBase_select_node =sixtracklib.st_ClContextBase_select_node
st_ClContextBase_select_node.argtypes = [st_Context_p, ct.c_char_p]
st_ClContextBase_select_node.restype  = None

st_ClContextBase_print_nodes_info = sixtracklib.st_ClContextBase_print_nodes_info
st_ClContextBase_print_nodes_info.argtypes = [st_Context_p]
st_ClContextBase_print_nodes_info.restype  = None

st_ClContextBase_delete = sixtracklib.st_ClContextBase_delete
st_ClContextBase_delete.argtypes = [ st_Context_p ]
st_ClContextBase_delete.restype  = None

