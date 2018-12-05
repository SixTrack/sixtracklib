from . import stcommon as st

class TrackJobCL(object):
    @classmethod
    def print_nodes(cls):
        context=st.st_ClContext_create()
        st.st_ClContextBase_print_nodes_info(context)

    def __init__(self,particles,elements,device="0.0"):
        self.particles=particles
        self.elements=elements
        self.context=st.st_ClContext_create()
        st.st_ClContextBase_select_node(self.context, device.encode("utf-8"))
        self.o_buff=st.st_NullBuffer
        self.particles_buffer=st.STBuffer(self.particles._buffer)

    def track(self,until_turn,elem_by_elem_turns=0):
        p_buff=st.STBuffer(self.particles._buffer)
        e_buff=st.STBuffer(self.elements.cbuffer)

        self.o_buff=st.st_TrackCL(self.context,
                                  p_buff.stbuffer,
                                  e_buff.stbuffer,
                                  self.o_buff,
                until_turn,elem_by_elem_turns)


