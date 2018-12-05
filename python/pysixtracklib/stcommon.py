import ctypes as ct

sixtracklib = ct.CDLL("/usr/local/lib/libsixtrack.so")

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

st_Context_p = ct.c_void_p

# C-API functions

st_Buffer_new_on_memory = sixtracklib.st_Buffer_new_on_memory
st_Buffer_new_on_memory.argtypes = [ct.c_char_p, ct.c_uint64]
st_Buffer_new_on_memory.restype = st_Buffer_p


st_Buffer_delete =  sixtracklib.st_Buffer_delete
st_Buffer_delete.argtypes = [st_Buffer_p]

st_TrackCL = sixtracklib.st_TrackCL
st_TrackCL.argtypes = [st_Context_p, st_Buffer_p, st_Buffer_p, st_Buffer_p, ct.c_uint64, ct.c_uint64]
st_TrackCL.restype = st_Buffer_p

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


