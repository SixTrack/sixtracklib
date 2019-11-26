import cobjects
from cobjects import CBuffer, CObject
import sixtracklib as st
from st.stcommon import st_ARCH_BEAM_ELEMENTS_BUFFER_ID, AssignAddressItem, \
    st_NullAssignAddressItem, st_AssignAddressItem_p, st_buffer_size, \
    st_object_type_id_t, st_arch_status_t, st_arch_size_t, \
    st_ARCH_STATUS_SUCCESS, st_ARCH_ILLEGAL_BUFFER_ID

if __name__ == '__main__':
    lattice = st.Elements()
    lattice.Drift(length=0.0)
    lattice.Drift(length=0.1)
    lattice.Drift(length=0.2)
    bm0_index = lattice.cbuffer.n_objects
    lattice.BeamMonitor()
    lattice.Drift(length=0.3)
    lattice.Drift(length=0.4)
    lattice.Drift(length=0.5)
    bm1_index = lattice.cbuffer.n_objects
    lattice.BeamMonitor()
    lattice.Drift(length=0.3)
    lattice.Drift(length=0.4)
    lattice.Drift(length=0.5)
    bm2_index = lattice.cbuffer.n_objects
    lattice.BeamMonitor()

    assert lattice.cbuffer.get_object(bm0_index).out_address == 0
    assert lattice.cbuffer.get_object(bm1_index).out_address == 0
    assert lattice.cbuffer.get_object(bm2_index).out_address == 0

    pset = st.ParticlesSet()
    pset.Particles(num_particles=100)

    output_buffer = st.ParticlesSet()
    out_buffer0_index = output_buffer.cbuffer.n_objects
    output_buffer.Particles(num_particles=100)
    out_buffer1_index = output_buffer.cbuffer.n_objects
    output_buffer.Particles(num_particls=512)

    job = st.TrackJob(lattice, pset)

    # hand the output_buffer over to the track job:
    output_buffer_id = job.add_stored_buffer(cbuffer=output_buffer)
    assert output_buffer_id != st_ARCH_ILLEGAL_BUFFER_ID.value

    # use the predefined lattice_buffer_id value to refer to the
    # beam elements buffer
    lattice_buffer_id = st_ARCH_BEAM_ELEMENTS_BUFFER_ID.value

    # use the _type_id attributes of beam monitors and particle sets to
    # refer to these object types:
    particle_set_type_id = output_buffer.cbuffer.get_object(
        out_buffer0_index)._type_id
    beam_monitor_type_id = lattice.cbuffer.get_object(bm0_index)._type_id

    assert job.total_num_assign_items == 0
    assert not job.has_assign_items( lattice_buffer_id, output_buffer_id )
    assert job.num_assign_items( lattice_buffer_id, output_buffer_id ) == 0

    # --------------------------------------------------------------------------
    # Create the assignment item for out_buffer0 -> bm0

    out0_to_bm0_addr_assign_item = AssignAddressItem(
        dest_elem_type_id=beam_monitor_type_id,
        dest_buffer_id=lattice_buffer_id,
        dest_elem_index=bm0_index,
        dest_pointer_offset=24, # Magic number, offset of out_address from begin
        src_elem_type_id=particle_set_type_id,
        src_buffer_id=output_buffer_id,
        src_elem_index=out_buffer0_index,
        src_pointer_offset=0 #We assign the starting address of the particle set
    )

    assert out0_to_bm0_addr_assign_item.dest_elem_type_id == \
            beam_monitor_type_id
    assert out0_to_bm0_addr_assign_item.dest_buffer_id == lattice_buffer_id
    assert out0_to_bm0_addr_assign_item.dest_elem_index == bm0_index
    assert out0_to_bm0_addr_assign_item.dest_pointer_offset == 24

    assert out0_to_bm0_addr_assign_item.src_elem_type_id == \
            particle_set_type_id
    assert out0_to_bm0_addr_assign_item.src_buffer_id == output_buffer_id
    assert out0_to_bm0_addr_assign_item.src_elem_index == out_buffer0_index
    assert out0_to_bm0_addr_assign_item.src_pointer_offset == 0

    # perform the assignment of assign_out0_to_bm0_item

    ptr_item_0_to_0 = job.add_assign_address_item( assign_out0_to_bm0 )
    assert ptr_item_0_to_0 != st_NullAssignAddressItem
    assert job.total_num_assign_items == 1
    assert job.has_assign_items( lattice_buffer_id, output_buffer_id )
    assert job.num_assign_items( lattice_buffer_id, output_buffer_id ) == 1
    assert job.has_assign_item( item=ptr_item_0_to_0 )
    assert job.has_assign_item( item=out0_to_bm0_addr_assign_item )
    assert job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm0_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer0_index, src_pointer_offset=0 )
    assert not job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm1_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer1_index, src_pointer_offset=0 )
    assert not job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm2_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer0_index, src_pointer_offset=0 )

    # --------------------------------------------------------------------------
    # Create the assignment item for out_buffer1 -> bm1 at the time of
    # passing it on to the track job:

    ptr_item_1_to_1 = job.add_assign_address_item(
        dest_elem_type_id=beam_monitor_type_id,
        dest_buffer_id=lattice_buffer_id,
        dest_elem_index=bm1_index,
        dest_pointer_offset=24,
        src_elem_type_id=particle_set_type_id,
        src_buffer_id=output_buffer_id,
        src_elem_index=out_buffer1_index,
        src_pointer_offset=0 )

    assert ptr_item1_to_1 != st_NullAssignAddressItem
    assert job.total_num_assign_items == 2
    assert job.has_assign_items( lattice_buffer_id, output_buffer_id )
    assert job.num_assign_items( lattice_buffer_id, output_buffer_id ) == 2
    assert job.has_assign_item( item=ptr_item_1_to_1 )
    assert not job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm2_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer0_index, src_pointer_offset=0 )

    # Create the assignment item for out_buffer0 -> bm2

    # Create a copy of out0_to_bm0_addr_assign_item on the same buffer
    # TODO: figure out a better way to do this?
    out0_to_bm2_addr_assign_item = AssignAddressItem(
        **{ k != '_buffer' and k or 'cbuffer': getattr(item0, k)
            for k in [ *[ f[ 0 ] for f in
                AssignAddressItem.get_fields() ], '_buffer' ]} )

    # out0_to_bm2_addr_assign_item is actually the same as
    # out0_to_bm0_addr_assign_item  -> if we try to add this item unmodified,
    # we should again effectively get ptr_item0_to_0:

    ptr_item_0_to_2 = job.add_assign_address_item(
            out0_to_bm2_addr_assign_item )

    assert ptr_item_0_to_2 != st_NullAssignAddressItem
    assert ptr_item_0_to_2 == ptr_item_0_to_0
    assert job.total_num_assign_items == 2
    assert job.has_assign_items( lattice_buffer_id, output_buffer_id )
    assert job.num_assign_items( lattice_buffer_id, output_buffer_id ) == 2
    assert job.has_assign_item( item=ptr_item_0_to_2 )
    assert not job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm2_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer0_index, src_pointer_offset=0 )


    #modify out0_to_bm2_addr_assign_item to target the third beam monitor
    #located at bm2_index:
    out0_to_bm2_addr_assign_item.dest_elem_index = bm2_index

    # try again to add -> this time it should result in a new item:
    ptr_item_0_to_2 = job.add_assign_address_item(
            out0_to_bm2_addr_assign_item )

    assert ptr_item_0_to_2 != st_NullAssignAddressItem
    assert ptr_item_0_to_2 != ptr_item_0_to_0
    assert job.total_num_assign_items == 3
    assert job.has_assign_items( lattice_buffer_id, output_buffer_id )
    assert job.num_assign_items( lattice_buffer_id, output_buffer_id ) == 3
    assert job.has_assign_item( item=out0_to_bm2_addr_assign_item )
    assert job.has_assign_item( item=ptr_item_0_to_2 )
    assert job.has_assign_item( dest_elem_type_id=beam_monitor_type_id, dest_buffer_id=lattice_buffer_id, dest_elem_index=bm2_index, dest_pointer_offset=24, src_elem_type_id=particle_set_type_id, src_buffer_id=output_buffer_id, src_elem_index=out_buffer0_index, src_pointer_offset=0 )


    # --------------------------------------------------------------------------
    # finish assembly of assign items:
    job.push_assign_address_items()

    # perform assignment of address items:
    job.perform_managed_assignments()

    job.collect_beam_elements()
    assert lattice.cbuffer.get_object(bm0_index).out_address != 0
    assert lattice.cbuffer.get_object(bm1_index).out_address != 0
    assert lattice.cbuffer.get_object(bm2_index).out_address != 0

    # this only works on the CPU:
    assert lattice.cbuffer.get_object(bm0_index).out_address == \
           output_buffer.get_object(out_buffer0_index)._get_address()
    assert lattice.cbuffer.get_object(bm1_index).out_address == \
           output_buffer.get_object(out_buffer1_index)._get_address()
    assert lattice.cbuffer.get_object(bm2_index).out_address == \
           output_buffer.get_object(out_buffer2_index)._get_address()
