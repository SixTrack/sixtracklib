#include "sixtracklib/opencl/track_job_cl.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer/assign_address_item.h"
#include "sixtracklib/common/be_drift/be_drift.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/context/compute_arch.h"
#include "sixtracklib/opencl/internal/base_context.h"

TEST( CXXOpenCLTrackJobClAssignAddressItemsTests, MinimalUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using track_job_t    = st::TrackJobCl;
    using size_t         = track_job_t::size_type;
    using assign_item_t  = track_job_t::assign_item_t;
    using buffer_t       = track_job_t::buffer_t;
    using c_buffer_t     = track_job_t::c_buffer_t;
    using controller_t   = track_job_t::context_t;
    using node_id_t      = controller_t::node_id_t;
    using status_t       = controller_t::status_t;
    using particle_set_t = st::Particles;
    using be_monitor_t   = st::BeamMonitor;
    using drift_t        = st::Drift;
    using addr_t         = st::buffer_addr_t;

    buffer_t my_lattice;
    buffer_t my_output_buffer;

    if( controller_t::NUM_AVAILABLE_NODES() == 0u )
    {
        std::cout << "No OpenCL nodes available -> skipping test\r\n";
        return;
    }

    st::Drift* dr = my_lattice.add< drift_t >( 0.0 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.1 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.2 );
    SIXTRL_ASSERT( dr != nullptr );

    size_t const bm0_elem_idx = my_lattice.getNumObjects();

    st::BeamMonitor* bm0 = my_lattice.createNew< be_monitor_t >();
    SIXTRL_ASSERT( bm0 != nullptr );
    ( void )bm0;

    dr = my_lattice.add< drift_t >( 0.0 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.1 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.2 );
    SIXTRL_ASSERT( dr != nullptr );

    size_t const bm1_elem_idx = my_lattice.getNumObjects();
    SIXTRL_ASSERT( bm1_elem_idx > bm0_elem_idx );

    st::BeamMonitor* bm1 = my_lattice.createNew< be_monitor_t >();
    SIXTRL_ASSERT( bm1 != nullptr );
    ( void )bm1;

    dr = my_lattice.add< drift_t >( 0.3 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.4 );
    SIXTRL_ASSERT( dr != nullptr );

    dr = my_lattice.add< drift_t >( 0.5 );
    SIXTRL_ASSERT( dr != nullptr );
    ( void )dr;

    size_t const bm2_elem_idx = my_lattice.getNumObjects();
    SIXTRL_ASSERT( bm2_elem_idx > bm1_elem_idx );

    st::BeamMonitor* bm2 = my_lattice.createNew< be_monitor_t >();
    SIXTRL_ASSERT( bm2 != nullptr );
    ( void )bm2;

    size_t const out_buffer0_index = my_output_buffer.getNumObjects();
    particle_set_t* out_buffer0 =
        my_output_buffer.createNew< st::Particles >( size_t{ 100 } );
    SIXTRL_ASSERT( out_buffer0 );

    size_t const out_buffer1_index = my_output_buffer.getNumObjects();
    particle_set_t* out_buffer1 =
        my_output_buffer.createNew< st::Particles >( size_t{ 512 } );
    SIXTRL_ASSERT( out_buffer1 );

    c_buffer_t* ptr_my_lattice_buffer = my_lattice.getCApiPtr();
    SIXTRL_ASSERT( ptr_my_lattice_buffer != nullptr );

    buffer_t::address_t const my_lattice_buffer_addr =
        my_lattice.getDataBeginAddr();

    size_t const my_lattice_buffer_size = my_lattice.getSize();
    size_t const my_lattice_buffer_capacity = my_lattice.getCapacity();

    size_t const my_lattice_buffer_num_objects =
        my_lattice.getNumObjects();

    c_buffer_t* ptr_my_output_buffer  = my_output_buffer.getCApiPtr();
    SIXTRL_ASSERT( ptr_my_output_buffer != nullptr );

    buffer_t::address_t const my_output_buffer_addr =
        my_output_buffer.getDataBeginAddr();

    size_t const my_output_buffer_size = my_output_buffer.getSize();
    size_t const my_output_buffer_capacity = my_output_buffer.getCapacity();

    size_t const my_output_buffer_num_objects =
        my_output_buffer.getNumObjects();


    out_buffer0 = my_output_buffer.get< particle_set_t >( out_buffer0_index );
    out_buffer1 = my_output_buffer.get< particle_set_t >( out_buffer1_index );

    /* --------------------------------------------------------------------- */

    node_id_t node_id;

    size_t const num_nodes = controller_t::GET_AVAILABLE_NODES(
        &node_id, size_t{ 1 } );
    ASSERT_TRUE( num_nodes == size_t{ 1 } );

    char NODE_ID_STR[ 32 ] =
    {
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
        '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
    };

    status_t status = ::NS(ComputeNodeId_to_string_with_format)(
        &node_id, &NODE_ID_STR[ 0 ], size_t{ 32 }, st::ARCHITECTURE_OPENCL,
            st::NODE_ID_STR_FORMAT_NOARCH );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    track_job_t job( std::string{ NODE_ID_STR } );

    ASSERT_TRUE( job.ptrContext() != nullptr );
    ASSERT_TRUE( job.ptrContext()->hasSelectedNode() );
    ASSERT_TRUE( job.ptrContext()->selectedNodeIdStr().compare(
        NODE_ID_STR ) == 0 );

    /* -------------------------------------------------------------------- */

    size_t const my_lattice_buffer_id = job.add_stored_buffer(
        std::move( my_lattice ) );

    ASSERT_TRUE( my_lattice_buffer_id != track_job_t::ILLEGAL_BUFFER_ID );

    size_t const my_output_buffer_id = job.add_stored_buffer(
        std::move( my_output_buffer ) );

    ptr_my_lattice_buffer = nullptr;
    ptr_my_output_buffer  = nullptr;

    ASSERT_TRUE( my_output_buffer_id != track_job_t::ILLEGAL_BUFFER_ID );
    ASSERT_TRUE( my_output_buffer_id > my_lattice_buffer_id );

    ASSERT_TRUE(  job.is_buffer_by_buffer_id( my_lattice_buffer_id ) );
    ASSERT_TRUE( !job.is_raw_memory_by_buffer_id( my_lattice_buffer_id ) );

    ptr_my_lattice_buffer = job.buffer_by_buffer_id( my_lattice_buffer_id );

    ASSERT_TRUE(  ptr_my_lattice_buffer != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_capacity );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_my_lattice_buffer ) ==
                    my_lattice_buffer_num_objects );

    ptr_my_output_buffer = job.buffer_by_buffer_id( my_output_buffer_id );

    ASSERT_TRUE( ptr_my_output_buffer != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_my_output_buffer ) ==
                    my_output_buffer_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_my_output_buffer ) ==
                    my_output_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_my_output_buffer ) ==
                    my_output_buffer_capacity );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ptr_my_output_buffer ) ==
                    my_output_buffer_num_objects );

    ASSERT_TRUE(  job.is_buffer_by_buffer_id( my_output_buffer_id ) );
    ASSERT_TRUE( !job.is_raw_memory_by_buffer_id( my_output_buffer_id ) );
    ASSERT_TRUE(  job.buffer_by_buffer_id( my_output_buffer_id ) ==
                  ptr_my_output_buffer );

    ASSERT_TRUE(  job.ptr_stored_buffer( my_output_buffer_id ) ==
                  ptr_my_output_buffer );

    assign_item_t const assign_item_0_to_0(
        st::OBJECT_TYPE_BEAM_MONITOR, my_lattice_buffer_id, bm0_elem_idx,
        offsetof( st::BeamMonitor, out_address ),
        st::OBJECT_TYPE_PARTICLE, my_output_buffer_id, out_buffer0_index,
        size_t{ 0 } );

    ASSERT_TRUE(  job.total_num_assign_items() == size_t{ 0 } );
    ASSERT_TRUE( !job.has_assign_address_item( assign_item_0_to_0 ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 0 } );

    ASSERT_TRUE( !job.has_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) );

    assign_item_t* ptr_assign_item_0_to_0 =
        job.add_assign_address_item( assign_item_0_to_0 );

    ASSERT_TRUE(  ptr_assign_item_0_to_0 != nullptr );
    ASSERT_TRUE(  assign_item_0_to_0 == *ptr_assign_item_0_to_0 );

    ASSERT_TRUE(  ptr_assign_item_0_to_0 != nullptr );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getDestElemTypeId() ==
                  st::OBJECT_TYPE_BEAM_MONITOR );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getDestBufferId() ==
                  my_lattice_buffer_id );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getDestElemIndex() ==
                  bm0_elem_idx );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getDestElemPointerOffset() ==
                  offsetof( st::BeamMonitor, out_address ) );

    ASSERT_TRUE(  ptr_assign_item_0_to_0->getSrcElemTypeId() ==
                  st::OBJECT_TYPE_PARTICLE );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getSrcBufferId() ==
                  my_output_buffer_id );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getSrcElemIndex() ==
                  out_buffer0_index );
    ASSERT_TRUE(  ptr_assign_item_0_to_0->getSrcElemPointerOffset() ==
                  size_t{ 0 } );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_addr_from_buffer)(
                    ptr_assign_item_0_to_0->getCApiPtr(),
                        ptr_my_output_buffer ) == reinterpret_cast< uintptr_t >(
                            out_buffer0 ) );

    ASSERT_TRUE(  job.has_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 1 } );

    ASSERT_TRUE(  job.total_num_assign_items() == size_t{ 1 } );

    ASSERT_TRUE( !job.has_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) == size_t{ 0 } );

    assign_item_t* ptr_assign_item_1_to_1 = job.add_assign_address_item(
        st::OBJECT_TYPE_BEAM_MONITOR, my_lattice_buffer_id, bm1_elem_idx,
        offsetof( st::BeamMonitor, out_address ),
        st::OBJECT_TYPE_PARTICLE, my_output_buffer_id, out_buffer1_index,
        size_t{ 0 } );

    ASSERT_TRUE(  ptr_assign_item_1_to_1 != nullptr );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getDestElemTypeId() ==
                  st::OBJECT_TYPE_BEAM_MONITOR );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getDestBufferId() ==
                  my_lattice_buffer_id );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getDestElemIndex() ==
                  bm1_elem_idx );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getDestElemPointerOffset() ==
                  offsetof( st::BeamMonitor, out_address ) );

    ASSERT_TRUE(  ptr_assign_item_1_to_1->getSrcElemTypeId() ==
                  st::OBJECT_TYPE_PARTICLE );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getSrcBufferId() ==
                  my_output_buffer_id );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getSrcElemIndex() ==
                  out_buffer1_index );
    ASSERT_TRUE(  ptr_assign_item_1_to_1->getSrcElemPointerOffset() ==
                  size_t{ 0 } );

    ASSERT_TRUE(  ::NS(AssignAddressItem_src_pointer_addr_from_buffer)(
                    ptr_assign_item_1_to_1->getCApiPtr(),
                        ptr_my_output_buffer ) == reinterpret_cast< uintptr_t >(
                            out_buffer1 ) );

    ASSERT_TRUE(  job.has_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 2 } );

    ASSERT_TRUE(  job.total_num_assign_items() == size_t{ 2 } );

    ASSERT_TRUE( !job.has_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) == size_t{ 0 } );

    assign_item_t assign_item_0_to_2( assign_item_0_to_0 );
    assign_item_0_to_2.dest_elem_index = bm2_elem_idx;

    assign_item_t* ptr_assign_item_0_to_2 =
        job.add_assign_address_item( assign_item_0_to_2 );

    ASSERT_TRUE(  ptr_assign_item_0_to_2 != nullptr );
    ASSERT_TRUE(  assign_item_0_to_2 == *ptr_assign_item_0_to_2 );

    ASSERT_TRUE(  job.has_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, my_output_buffer_id ) == size_t{ 3 } );

    ASSERT_TRUE(  job.total_num_assign_items() == size_t{ 3 } );

    ASSERT_TRUE( !job.has_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) );

    ASSERT_TRUE(  job.num_assign_items(
        my_lattice_buffer_id, st::ARCH_OUTPUT_BUFFER_ID ) == size_t{ 0 } );

    status = job.commit_address_assignments();
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = job.assign_all_addresses();
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    status = job.collect_stored_buffer( my_lattice_buffer_id );
    ASSERT_TRUE(status == st::ARCH_STATUS_SUCCESS );

    bm0 = reinterpret_cast< be_monitor_t* >( ::NS(BeamMonitor_from_buffer)(
        ptr_my_lattice_buffer, bm0_elem_idx ) );

    bm1 = reinterpret_cast< be_monitor_t* >( ::NS(BeamMonitor_from_buffer)(
        ptr_my_lattice_buffer, bm1_elem_idx ) );

    bm2 = reinterpret_cast< be_monitor_t* >( ::NS(BeamMonitor_from_buffer)(
        ptr_my_lattice_buffer, bm2_elem_idx ) );

    SIXTRL_ASSERT( bm0 != nullptr );
    SIXTRL_ASSERT( bm1 != nullptr );
    SIXTRL_ASSERT( bm2 != nullptr );

    ASSERT_TRUE( bm0->out_address != addr_t{ 0 } );
    ASSERT_TRUE( bm1->out_address != addr_t{ 0 } );
    ASSERT_TRUE( bm2->out_address != addr_t{ 0 } );

    ASSERT_TRUE( bm0->out_address == bm2->out_address );
    ASSERT_TRUE( bm1->out_address != bm0->out_address );
    ASSERT_TRUE( bm0->out_address <  bm1->out_address );

    ASSERT_TRUE( ( bm1->out_address - bm0->out_address ) ==
        static_cast< addr_t >(
            reinterpret_cast< uintptr_t >( out_buffer1 ) -
            reinterpret_cast< uintptr_t >( out_buffer0 ) ) );
}
