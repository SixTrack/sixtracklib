import cobjects
from cobjects import CBuffer, CObject
import sixtracklib as st

if __name__ == '__main__':
    lattice = st.Elements()
    lattice.Drift(length=1.0)
    pset = st.ParticlesSet()
    pset.Particles(num_particles=100)

    job = st.TrackJob(lattice, pset)

    assert not job.has_stored_buffers
    assert job.num_stored_buffers == 0
    assert job.min_stored_buffer_id == 0
    assert job.max_stored_buffer_id == 0

    data_buffer = st.ParticlesSet()
    out_data_index = data_buffer.cbuffer.n_objects
    out_data = data_buffer.Particles(num_particles=1000)
    assert data_buffer.cbuffer.n_objects == 1
    assert data_buffer.cbuffer.size > 0

    data_buffer_id = job.add_stored_buffer(cbuffer=data_buffer)
    assert data_buffer_id != st.stcommon.st_ARCH_ILLEGAL_BUFFER_ID.value
    assert job.has_stored_buffers
    assert job.num_stored_buffers == 1
    assert job.min_stored_buffer_id == data_buffer_id
    assert job.max_stored_buffer_id == data_buffer_id

    st_data_buffer = job.stored_buffer(data_buffer_id)
    assert st_data_buffer.pointer != st.stcommon.st_NullBuffer
    assert st_data_buffer.num_objects == data_buffer.cbuffer.n_objects
    assert st_data_buffer.size == data_buffer.cbuffer.size

    ptr_data_buffer = job.ptr_stored_buffer(data_buffer_id)
    assert ptr_data_buffer != st.stcommon.st_NullBuffer
    assert st.stcommon.st_Buffer_get_size(
        ptr_data_buffer) == data_buffer.cbuffer.size
    assert st.stcommon.st_Buffer_get_num_of_objects(
        ptr_data_buffer) == data_buffer.cbuffer.n_objects
