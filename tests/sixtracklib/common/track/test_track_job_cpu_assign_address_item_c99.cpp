#include "sixtracklib/common/track_job_cpu.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/buffer/assign_address_item.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/particles.h"


TEST( C99_Cpu_CpuTrackJob_AssignAddressItemTests, MinimalUsage )
{
    using track_job_t    = ::NS(TrackJobCpu);
    using size_t         = ::NS(arch_size_t);
    using assign_item_t  = ::NS(AssignAddressItem);
    using c_buffer_t     = ::NS(Buffer);
    using particle_set_t = ::NS(Particles);

    c_buffer_t* my_lattice = ::NS(Buffer_new)( size_t{ 0 } );
    c_buffer_t* my_output_buffer = ::NS(Buffer_new)( size_t{ 0 } );

    ::NS(Drift)* dr = ::NS(Drift_add)( my_lattice, 0.0 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.1 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.2 );
    SIXTRL_ASSERT( dr != nullptr );

    size_t const bm0_elem_idx = ::NS(Buffer_get_num_of_objects)( my_lattice );

    ::NS(BeamMonitor)* bm0 = ::NS(BeamMonitor_new)( my_lattice );
    SIXTRL_ASSERT( bm0 != nullptr );
    ( void )bm0;

    dr = ::NS(Drift_add)( my_lattice, 0.0 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.1 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.2 );
    SIXTRL_ASSERT( dr != nullptr );

    size_t const bm1_elem_idx = ::NS(Buffer_get_num_of_objects)( my_lattice );
    SIXTRL_ASSERT( bm1_elem_idx > bm0_elem_idx );

    ::NS(BeamMonitor)* bm1 = ::NS(BeamMonitor_new)( my_lattice );
    SIXTRL_ASSERT( bm1 != nullptr );
    ( void )bm1;

    dr = ::NS(Drift_add)( my_lattice, 0.3 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.4 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = ::NS(Drift_add)( my_lattice, 0.5 );
    SIXTRL_ASSERT( dr != nullptr );
    ( void )dr;

    size_t const bm2_elem_idx = ::NS(Buffer_get_num_of_objects)( my_lattice );
    SIXTRL_ASSERT( bm2_elem_idx > bm1_elem_idx );

    ::NS(BeamMonitor)* bm2 = ::NS(BeamMonitor_new)( my_lattice );
    SIXTRL_ASSERT( bm2 != nullptr );
    ( void )bm2;

    size_t const out_buffer0_index =
        ::NS(Buffer_get_num_of_objects)( my_output_buffer );

    particle_set_t* out_buffer0 =
        ::NS(Particles_new)( my_output_buffer, size_t{ 100 } );
    SIXTRL_ASSERT( out_buffer0 );

    size_t const out_buffer1_index =
        ::NS(Buffer_get_num_of_objects)( my_output_buffer );

    particle_set_t* out_buffer1 =
        ::NS(Particles_new)( my_output_buffer, size_t{ 512 } );
    SIXTRL_ASSERT( out_buffer1 );

    ::NS(buffer_addr_t) const my_lattice_buffer_addr =
        ::NS(Buffer_get_data_begin_addr)( my_lattice );

    size_t const my_lattice_buffer_size =
        ::NS(Buffer_get_size)( my_lattice );

    size_t const my_lattice_buffer_capacity =
        ::NS(Buffer_get_capacity)( my_lattice );

    size_t const my_lattice_buffer_num_objects =
        ::NS(Buffer_get_num_of_objects)( my_lattice );

    ::NS(buffer_addr_t) const my_output_buffer_addr =
        ::NS(Buffer_get_data_begin_addr)( my_output_buffer );

    size_t const my_output_buffer_size =
        ::NS(Buffer_get_size)( my_output_buffer );

    size_t const my_output_buffer_capacity =
        ::NS(Buffer_get_capacity)( my_output_buffer );

    size_t const my_output_buffer_num_objects =
        ::NS(Buffer_get_num_of_objects)( my_output_buffer );


    out_buffer0 = ::NS(Particles_buffer_get_particles)(
        my_output_buffer, out_buffer0_index );

    out_buffer1 = ::NS(Particles_buffer_get_particles)(
        my_output_buffer, out_buffer1_index );

    track_job_t* job = ::NS(TrackJobCpu_create)();

    size_t const my_lattice_buffer_id = ::NS(TrackJob_add_stored_buffer)(
        job, my_lattice, false, false );

    ASSERT_TRUE( my_lattice_buffer_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    size_t const my_output_buffer_id = ::NS(TrackJob_add_stored_buffer)(
        job, my_output_buffer, false, false );

    c_buffer_t* ptr_my_lattice_buffer = nullptr;
    c_buffer_t* ptr_my_output_buffer  = nullptr;

    ASSERT_TRUE( my_output_buffer_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );
    ASSERT_TRUE( my_output_buffer_id > my_lattice_buffer_id );

    ASSERT_TRUE(  ::NS(TrackJob_is_buffer_by_buffer_id)(
        job, my_lattice_buffer_id ) );

    ASSERT_TRUE( !::NS(TrackJob_is_raw_memory_by_buffer_id)(
        job, my_lattice_buffer_id ) );

    ptr_my_lattice_buffer = ::NS(TrackJob_buffer_by_buffer_id)(
        job, my_lattice_buffer_id );

    ASSERT_TRUE(  ptr_my_lattice_buffer != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_capacity );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_num_objects );

    ptr_my_output_buffer = ::NS(TrackJob_buffer_by_buffer_id)(
        job, my_output_buffer_id );

    ASSERT_TRUE( ptr_my_output_buffer != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_my_output_buffer ) ==
                    my_output_buffer_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_my_output_buffer ) ==
                    my_output_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_my_output_buffer ) ==
                    my_output_buffer_capacity );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_my_output_buffer ) ==
                    my_output_buffer_num_objects );

    ASSERT_TRUE(  ::NS(TrackJob_is_buffer_by_buffer_id)(
        job, my_output_buffer_id ) );

    ASSERT_TRUE( !::NS(TrackJob_is_raw_memory_by_buffer_id)(
        job, my_output_buffer_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_buffer_by_buffer_id)(
        job, my_output_buffer_id ) == ptr_my_output_buffer );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)( job, my_output_buffer_id ) ==
        ptr_my_output_buffer );

    /* --------------------------------------------------------------------- */

    assign_item_t assign_item_0_to_0;
    ::NS(AssignAddressItem_preset)( &assign_item_0_to_0 );

    ::NS(AssignAddressItem_set_dest_elem_type_id)(
        &assign_item_0_to_0, ::NS(OBJECT_TYPE_BEAM_MONITOR) );

    ::NS(AssignAddressItem_set_dest_buffer_id)(
        &assign_item_0_to_0, my_lattice_buffer_id );

    ::NS(AssignAddressItem_set_dest_elem_index)(
        &assign_item_0_to_0, bm0_elem_idx );

    ::NS(AssignAddressItem_set_dest_pointer_offset)(
        &assign_item_0_to_0, offsetof( ::NS(BeamMonitor), out_address ) );

    ::NS(AssignAddressItem_set_src_elem_type_id)(
        &assign_item_0_to_0, ::NS(OBJECT_TYPE_PARTICLE) );

    ::NS(AssignAddressItem_set_src_buffer_id)(
        &assign_item_0_to_0, my_output_buffer_id );

    ::NS(AssignAddressItem_set_src_elem_index)(
        &assign_item_0_to_0, out_buffer0_index );

    ::NS(AssignAddressItem_set_src_pointer_offset)(
        &assign_item_0_to_0, size_t{ 0 } );


    ASSERT_TRUE(  ::NS(TrackJob_total_num_assign_items)( job ) == size_t{ 0 } );
    ASSERT_TRUE( !::NS(TrackJob_has_assign_address_item)(
        job, &assign_item_0_to_0 ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 0 } );

    ASSERT_TRUE( !::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );


    assign_item_t* ptr_assign_item_0_to_0 =
        ::NS(TrackJob_add_assign_address_item)( job, &assign_item_0_to_0 );

    ASSERT_TRUE(  ptr_assign_item_0_to_0 != nullptr );
    ASSERT_TRUE(  ::NS(AssignAddressItem_are_equal)(
                        &assign_item_0_to_0, ptr_assign_item_0_to_0 ) );

    ASSERT_TRUE(  ptr_assign_item_0_to_0 != nullptr );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_elem_type_id)(
                    ptr_assign_item_0_to_0 ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_buffer_id)(
                    ptr_assign_item_0_to_0 ) == my_lattice_buffer_id );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_elem_index)(
                    ptr_assign_item_0_to_0 ) == bm0_elem_idx );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_pointer_offset)(
                    ptr_assign_item_0_to_0 ) ==
                    offsetof( ::NS(BeamMonitor), out_address ) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_elem_type_id)(
                    ptr_assign_item_0_to_0 ) == ::NS(OBJECT_TYPE_PARTICLE) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_buffer_id)(
                    ptr_assign_item_0_to_0 ) == my_output_buffer_id );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_elem_index)(
                    ptr_assign_item_0_to_0 ) == out_buffer0_index );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_offset)(
                    ptr_assign_item_0_to_0 ) == size_t{ 0 } );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_addr_from_buffer)(
                        ptr_assign_item_0_to_0, ptr_my_output_buffer ) ==
                            reinterpret_cast< uintptr_t >( out_buffer0 ) );


    ASSERT_TRUE(  ::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 1 } );

    ASSERT_TRUE(  ::NS(TrackJob_total_num_assign_items)( job ) == size_t{ 1 } );

    ASSERT_TRUE( !::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) == size_t{ 0 } );

    size_t const item_0_to_0_index = ::NS(TrackJob_index_of_assign_address_item)(
        job, ptr_assign_item_0_to_0 );

    ASSERT_TRUE( item_0_to_0_index < ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );

    assign_item_t assign_item_1_to_1;
    ::NS(AssignAddressItem_preset)( &assign_item_1_to_1 );

    ::NS(AssignAddressItem_set_dest_elem_type_id)(
        &assign_item_1_to_1, ::NS(OBJECT_TYPE_BEAM_MONITOR) );

    ::NS(AssignAddressItem_set_dest_buffer_id)(
        &assign_item_1_to_1, my_lattice_buffer_id );

    ::NS(AssignAddressItem_set_dest_elem_index)(
        &assign_item_1_to_1, bm1_elem_idx );

    ::NS(AssignAddressItem_set_dest_pointer_offset)(
        &assign_item_1_to_1, offsetof( ::NS(BeamMonitor), out_address ) );

    ::NS(AssignAddressItem_set_src_elem_type_id)(
        &assign_item_1_to_1, ::NS(OBJECT_TYPE_PARTICLE) );

    ::NS(AssignAddressItem_set_src_buffer_id)(
        &assign_item_1_to_1, my_output_buffer_id );

    ::NS(AssignAddressItem_set_src_elem_index)(
        &assign_item_1_to_1, out_buffer1_index );

    ::NS(AssignAddressItem_set_src_pointer_offset)(
        &assign_item_1_to_1, size_t{ 0 } );

    ASSERT_TRUE( ::NS(TrackJob_ptr_assign_address_item)(
        job, &assign_item_1_to_1 ) == nullptr );

    ASSERT_TRUE( ::NS(TrackJob_ptr_assign_address_item_detailed)(
        job, ::NS(OBJECT_TYPE_BEAM_MONITOR), my_lattice_buffer_id,
            bm1_elem_idx, offsetof( ::NS(BeamMonitor), out_address ),
                ::NS(OBJECT_TYPE_PARTICLE), my_output_buffer_id,
                    out_buffer1_index, size_t{ 0 } ) == nullptr );

    /* --------------------------------------------------------------------- */

    assign_item_t* ptr_assign_item_1_to_1 =
    ::NS(TrackJob_add_assign_address_item_detailed)( job,
        ::NS(OBJECT_TYPE_BEAM_MONITOR), my_lattice_buffer_id, bm1_elem_idx,
            offsetof( ::NS(BeamMonitor), out_address ),
                ::NS(OBJECT_TYPE_PARTICLE), my_output_buffer_id,
                    out_buffer1_index, size_t{ 0 } );

    ASSERT_TRUE(  ptr_assign_item_1_to_1 != nullptr );
    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_elem_type_id)(
                    ptr_assign_item_1_to_1 ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_buffer_id)(
                    ptr_assign_item_1_to_1 ) == my_lattice_buffer_id );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_elem_index)(
                    ptr_assign_item_1_to_1 ) == bm1_elem_idx );

    ASSERT_TRUE(  ::NS(AssignAddressItem_dest_pointer_offset)(
                    ptr_assign_item_1_to_1 ) == offsetof(
                        ::NS(BeamMonitor), out_address ) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_elem_type_id)(
                    ptr_assign_item_1_to_1 ) == ::NS(OBJECT_TYPE_PARTICLE) );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_buffer_id)(
                    ptr_assign_item_1_to_1 ) == my_output_buffer_id );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_elem_index)(
                    ptr_assign_item_1_to_1 ) == out_buffer1_index );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_offset)(
                    ptr_assign_item_1_to_1 ) == size_t{ 0 } );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_addr_from_buffer)(
                        ptr_assign_item_1_to_1, ptr_my_output_buffer ) ==
                            reinterpret_cast< uintptr_t >( out_buffer1 ) );


    ASSERT_TRUE(  ::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 2 } );

    ASSERT_TRUE(  ::NS(TrackJob_total_num_assign_items)(
        job ) == size_t{ 2 } );

    ASSERT_TRUE( !::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) == size_t{ 0 } );

    size_t const item_1_to_1_index = ::NS(TrackJob_index_of_assign_address_item)(
        job, ptr_assign_item_1_to_1 );

    ASSERT_TRUE( item_1_to_1_index < ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );

    /* --------------------------------------------------------------------- */

    assign_item_t assign_item_0_to_2 = assign_item_0_to_0;

    ::NS(AssignAddressItem_set_dest_elem_index)(
        &assign_item_0_to_2, bm2_elem_idx );

    assign_item_t* ptr_assign_item_0_to_2 =
        ::NS(TrackJob_add_assign_address_item)( job, &assign_item_0_to_2 );

    ASSERT_TRUE(  ptr_assign_item_0_to_2 != nullptr );
    ASSERT_TRUE(  ::NS(AssignAddressItem_are_equal)(
        &assign_item_0_to_2, ptr_assign_item_0_to_2 ) );

    ASSERT_TRUE(  ::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 3 } );

    ASSERT_TRUE(  ::NS(TrackJob_total_num_assign_items)(
        job ) == size_t{ 3 } );

    ASSERT_TRUE( !::NS(TrackJob_has_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_assign_items)( job,
        my_lattice_buffer_id, ::NS(ARCH_OUTPUT_BUFFER_ID) ) == size_t{ 0 } );

    /* --------------------------------------------------------------------- */

    ::NS(arch_status_t) status =
        ::NS(TrackJob_perform_all_managed_assignments)( job );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    bm0 = ::NS(BeamElements_buffer_get_beam_monitor)(
                ptr_my_lattice_buffer, bm0_elem_idx );

    bm1 = ::NS(BeamElements_buffer_get_beam_monitor)(
                ptr_my_lattice_buffer, bm1_elem_idx );

    bm2 = ::NS(BeamElements_buffer_get_beam_monitor)(
                ptr_my_lattice_buffer, bm2_elem_idx );

    SIXTRL_ASSERT( bm0 != nullptr );
    SIXTRL_ASSERT( bm1 != nullptr );
    SIXTRL_ASSERT( bm2 != nullptr );

    ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( bm0 ) ==
                        reinterpret_cast< uintptr_t >( out_buffer0 ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( bm1 ) ==
                        reinterpret_cast< uintptr_t >( out_buffer1 ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( bm2 ) ==
                        reinterpret_cast< uintptr_t >( out_buffer0 ) );

    /* --------------------------------------------------------------------- */

    ::NS(TrackJob_delete)( job );
    ::NS(Buffer_delete)( my_lattice );
    ::NS(Buffer_delete)( my_output_buffer );
}
