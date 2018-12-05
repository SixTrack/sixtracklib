from .stcommon import STBuffer, st_TrackCL, st_NullBuffer, st_NullChar


class TrackJobCL(object):
    @classmethod
    def print_nodes(cls):
        st_TrackCL(st_NullChar, st_NullBuffer, st_NullBuffer, st_NullBuffer,0,0)

    def __init__(self,particles,elements,device="0.0",output=None):
        self.particles=particles
        self.elements=elements
        self.device=device
        self.output=output

    def track(self,until_turn,elem_by_elem_turns=0):
        p_buff=STBuffer(self.particles._buffer).stbuffer
        e_buff=STBuffer(self.elements.cbuffer).stbuffer

        if self.output is None:
            o_buff=st_NullBuffer
        else:
            o_buff=self.output

        self.output=st_TrackCL(self.device.encode('utf-8'),p_buff,e_buff,o_buff,
                until_turn,elem_by_elem_turns)




