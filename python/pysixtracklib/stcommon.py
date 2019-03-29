import ctypes as ct
from .pysixtracklib import SHARED_LIB as st_SHARED_LIB

sixtracklib = ct.CDLL( st_SHARED_LIB )

# C-API Types

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

st_NullBuffer=ct.cast(0,st_Buffer_p)
st_Null=ct.cast(0,ct.c_void_p)
st_NullChar=ct.cast(0,ct.c_char_p)

st_Context_p  = ct.c_void_p
st_TrackJob_p = ct.c_void_p;

# C-API functions

st_Buffer_new_on_memory = sixtracklib.st_Buffer_new_on_memory
st_Buffer_new_on_memory.argtypes = [ct.c_char_p, ct.c_uint64]
st_Buffer_new_on_memory.restype = st_Buffer_p


st_Buffer_delete =  sixtracklib.st_Buffer_delete
st_Buffer_delete.argtypes = [st_Buffer_p]

# -----------------------------------------------------------------------------
# TrackJobCpu objects

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


st_TrackJob_track  = sixtracklib.st_TrackJob_track_until_turn
st_TrackJob_track.argtypes  = [ st_TrackJob_p, ct.c_uint64 ]
st_TrackJob_track.restype   = ct.c_int32


st_TrackJob_track_elem_by_elem  = sixtracklib.st_TrackJob_track_until_turn
st_TrackJob_track_elem_by_elem.argtypes  = [ st_TrackJob_p, ct.c_uint64 ]
st_TrackJob_track_elem_by_elem.restype   = ct.c_int32


st_TrackJob_collect = sixtracklib.st_TrackJob_collect
st_TrackJob_collect.argtypes = [ st_TrackJob_p ]


st_TrackJob_get_type_id = sixtracklib.st_TrackJob_get_type_id
st_TrackJob_get_type_id.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_type_id.restype  = ct.c_int64


st_TrackJob_get_type_str = sixtracklib.st_TrackJob_get_type_str
st_TrackJob_get_type_str.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_type_str.restype  = ct.c_char_p


st_TrackJob_has_output_buffer = sixtracklib.st_TrackJob_has_output_buffer
st_TrackJob_has_output_buffer.argtypes = [ st_TrackJob_p ]
st_TrackJob_has_output_buffer.restype  = ct.c_bool


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


st_TrackJob_get_num_beam_monitor = sixtracklib.st_TrackJob_get_num_beam_monitor
st_TrackJob_get_num_beam_monitor.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_num_beam_monitor.restype  = ct.c_uint64


st_TrackJob_get_beam_monitor_output_buffer_offset = \
    sixtracklib.st_TrackJob_get_beam_monitor_output_buffer_offset
st_TrackJob_get_beam_monitor_output_buffer_offset.argtypes = [ st_TrackJob_p ]
st_TrackJob_get_beam_monitor_output_buffer_offset.restype  = ct.c_uint64


# -----------------------------------------------------------------------------

st_ClContext_create = sixtracklib.st_ClContext_create
st_ClContext_create.restype = st_Context_p

st_ClContextBase_select_node =sixtracklib.st_ClContextBase_select_node
st_ClContextBase_select_node.argtypes = [st_Context_p, ct.c_char_p]

st_ClContextBase_print_nodes_info = sixtracklib.st_ClContextBase_print_nodes_info
st_ClContextBase_print_nodes_info.argtypes = [st_Context_p]

st_ClContextBase_delete = sixtracklib.st_ClContextBase_delete

# Helper Classes

class STBuffer(object):
    def __init__(self, cbuffer):
        self.cbuffer = cbuffer
        buff = ct.cast(self.cbuffer.base, ct.c_char_p)
        size = self.cbuffer.size
        self.stbuffer = st_Buffer_new_on_memory(buff, size)

    def info(self):
        print(self.stbuffer.contents)

    def __del__(self):
        print(f"De-allocate STBuffer {self.stbuffer}")
        st_Buffer_delete(self.stbuffer)


