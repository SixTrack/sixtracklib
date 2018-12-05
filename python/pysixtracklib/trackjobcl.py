from . import stcommon as st

class TrackJobCL(object):
    @classmethod
    def print_nodes(cls):
        context=st.st_ClContext_create()
        st.st_ClContextBase_print_nodes_info(context)

    def __init__(self,particles,elements,device="0.0",output=None):
        self.particles=particles
        self.elements=elements
        self.output=output
        self.context=st.st_ClContext_create()
        st.st_ClContextBase_select_node(self.context, device.encode("utf-8"))

    def track(self,until_turn,elem_by_elem_turns=0):
        p_buff=st.STBuffer(self.particles._buffer).stbuffer
        e_buff=st.STBuffer(self.elements.cbuffer).stbuffer

        if self.output is None:
            o_buff=st.st_NullBuffer
        else:
            o_buff=self.output

        self.output=st.st_TrackCL(self.context,p_buff,e_buff,o_buff,
                until_turn,elem_by_elem_turns)


