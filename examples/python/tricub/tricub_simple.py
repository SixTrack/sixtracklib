import sixtracklib as st
from sixtracklib.tricub import TriCub, TriCubData, TriCub_buffer_create_assign_address_item
import cobjects
from cobjects import CBuffer

# First: build the lattice
lattice = CBuffer()
elem = st.Drift(cbuffer=lattice, length=1.0)
elem = st.MultiPole(cbuffer=lattice, knl=[0.0, 1.0] )
tc2  = TriCub(cbuffer=lattice) # First TriCub element: index 2
elem = st.Drift(cbuffer=lattice, length=2.0)
elem = st.LimitRect(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
tc5  = TriCub(cbuffer=lattice) # Second TriCub element: index 5
elem = st.Drift(cbuffer=lattice, length=0.5)
elem = st.LimitRect(xmin=-0.5, xmax=1.5, ymin=-1.5, ymax=0.5)
tc8  = TriCub(cbuffer=lattice) # Second TriCub element: index 8

# Second: create the data buffer and put two data elements on it
data_buffer = CBuffer()
tcdata0 = TriCubData(cbuffer=data_buffer, nx=100, ny=100, nz=100)
tcdata1 = TriCubData(cbuffer=data_buffer, nx=10, ny=16, nz=8)

# Third: Create the assignment table which keeps track of which
# TriCubData element is assigned to which TriCub beam element
assignment_buffer = CBuffer()

assign0_2 = TriCub_buffer_create_assign_address_item(
    assignment_buffer, lattice, 2, data_buffer, 0 )

assign1_5 = TriCub_buffer_create_assign_address_item(
    assignment_buffer, lattice, 5, data_buffer, 1 )

assign0_8 = TriCub_buffer_create_assign_address_item(
    assignment_buffer, lattice, 8, data_buffer, 0 )

# For OpenCL:
