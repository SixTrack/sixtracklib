import cobjects
from cobjects import CBuffer, CObject
import sixtracklib as st
from sixtracklib.tricub import TriCub, TriCubData
import pdb

if __name__ == '__main__':
    lattice = st.Elements()
    lattice.Drift(length=1.0)
    pset = st.ParticlesSet()
    pset.Particles(num_particles=100)

    job = st.TrackJob(lattice, pset)

    assert not job.has_ext_stored_buffers
    assert job.num_ext_stored_buffers == 0
    assert job.min_ext_stored_buffer_id == 0
    assert job.max_ext_stored_buffer_id == 0

    data_buffer = CBuffer()
    tricub_data = TriCubData(cbuffer=data_buffer, nx=1000, ny=1000, nz=100)
    assert data_buffer.n_objects == 1
    assert data_buffer.size > 0

    data_buffer_id = job.add_ext_stored_buffer(cbuffer=data_buffer)
    assert data_buffer_id != st.stcommon.st_ARCH_ILLEGAL_BUFFER_ID.value
    assert job.has_ext_stored_buffers
    assert job.num_ext_stored_buffers == 1
    assert job.min_ext_stored_buffer_id == data_buffer_id
    assert job.max_ext_stored_buffer_id == data_buffer_id

    st_data_buffer = job.ext_stored_st_buffer(data_buffer_id)
    assert st_data_buffer.pointer != st.stcommon.st_NullBuffer
    assert st_data_buffer.num_objects == data_buffer.n_objects
    assert st_data_buffer.size == data_buffer.size

    ptr_data_buffer = job.ptr_ext_stored_buffer(data_buffer_id)
    assert ptr_data_buffer != st.stcommon.st_NullBuffer
    assert st.stcommon.st_Buffer_get_size(
        ptr_data_buffer) == data_buffer.size
    assert st.stcommon.st_Buffer_get_num_of_objects(
        ptr_data_buffer) == data_buffer.n_objects
