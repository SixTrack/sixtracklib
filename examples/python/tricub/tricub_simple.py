import sixtracklib as st
import cobjects
from cobjects import CBuffer

# ------------------------------------------------------------------------------
# 1) Build the lattice and the particle set:
# a) the lattice
lattice = st.Elements()
elem = lattice.Drift(length=1.0)
elem = lattice.Multipole(knl=[0.0, 1.0])

tc2_index = lattice.cbuffer.n_objects  # First TriCub element: index 2
tc = st.TriCub(cbuffer=lattice.cbuffer)

elem = lattice.Drift(length=2.0)
elem = lattice.LimitRect(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)

tc5_index = lattice.cbuffer.n_objects  # Second TriCub element: index 5
tc = st.TriCub(cbuffer=lattice.cbuffer)

elem = lattice.Drift(length=0.5)
elem = lattice.LimitRect(xmin=-0.5, xmax=1.5, ymin=-1.5, ymax=0.5)

tc8_index = lattice.cbuffer.n_objects  # Third TriCub element: index 8
tc = st.TriCub(cbuffer=lattice.cbuffer)

# b) the particle set
particle_sets = st.ParticlesSet()
particles = particle_sets.Particles(num_particles=100)

# ------------------------------------------------------------------------------
# 2) Create the track_job; currently only CPU is supported

job = st.TrackJob(lattice, particle_sets)

# ------------------------------------------------------------------------------
# 3) Create the data buffer for the TriCubData instances and hand it over to
#    the track_job for management:

tricub_data_buffer = CBuffer()

tc_data_0_index = tricub_data_buffer.n_objects
tc_data_0 = st.TriCubData(cbuffer=tricub_data_buffer, nx=100, ny=100, nz=100)

tc_data_1_index = tricub_data_buffer.n_objects
tc_data_1 = st.TriCubData(cbuffer=tricub_data_buffer, nx=10, ny=16, nz=8)

tricub_data_buffer_id = job.add_stored_buffer(cbuffer=tricub_data_buffer)

# ------------------------------------------------------------------------------
# 4) Create the mappings connecting the two TriCubData instances to the three
#    TriCub beam elements
#
#    tc_data_0 -> tc2
#    tc_data_1 -> tc5
#    tc_data_0 -> tc8

st.TriCub_buffer_create_assign_address_item(
    job, tc2_index, tricub_data_buffer_id, tc_data_0_index)
st.TriCub_buffer_create_assign_address_item(
    job, tc5_index, tricub_data_buffer_id, tc_data_1_index)
st.TriCub_buffer_create_assign_address_item(
    job, tc8_index, tricub_data_buffer_id, tc_data_0_index)

# commit the mappings:

job.commit_address_assignments()

# ------------------------------------------------------------------------------
# 5) Perform the assignments

job.assign_all_addresses()

# ------------------------------------------------------------------------------
# 6) Check whether the assignments actually worked

job.collect_beam_elements()
job.collect_stored_buffer(tricub_data_buffer_id)

tc2 = job.beam_elements_buffer.get_object(tc2_index)
tc5 = job.beam_elements_buffer.get_object(tc5_index)
tc8 = job.beam_elements_buffer.get_object(tc8_index)

if job.arch_str == 'cpu':
    tc2_data_addr = tc2.data_addr
    tc5_data_addr = tc5.data_addr
    tc8_data_addr = tc8.data_addr

    tc_data_0_addr = tc_data_0._get_address()
    tc_data_1_addr = tc_data_1._get_address()

    print_str = f"""
    tc2.data_addr = {tc2_data_addr:#018x} <- tc_data_0 @ {tc_data_0_addr:#018x}
    tc5.data_addr = {tc5_data_addr:#018x} <- tc_data_1 @ {tc_data_1_addr:#018x}
    tc8.data_addr = {tc8_data_addr:#018x} <- tc_data_0 @ {tc_data_0_addr:#018x}
    """
    print(print_str)
