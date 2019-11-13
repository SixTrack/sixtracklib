#include "sixtracklib/common/buffer/assign_address_item.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/internal/objects_type_id.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"

#include "sixtracklib/testlib.h"

TEST( C99_Common_Buffer_AssignAddressItemTests, BeamMonitorAssignment )
{
    using be_monitor_t   = ::NS(BeamMonitor);
    using buffer_t       = ::NS(Buffer);
    using buf_size_t     = ::NS(buffer_size_t);
    using assign_item_t  = ::NS(AssignAddressItem);
    using particle_set_t = ::NS(Particles);

    buffer_t* map_buffer    = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* beam_elements = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* output_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    constexpr buf_size_t NUM_BEAM_MONITORS = buf_size_t{ 10 };

    std::vector< buf_size_t > be_mon_indices(
        NUM_BEAM_MONITORS, buf_size_t{ 0 } );
    be_mon_indices.clear();

    std::vector< buf_size_t > out_buffer_indices(
        NUM_BEAM_MONITORS,  buf_size_t{ 0 } );

    particle_set_t* pset_dummy = ::NS(Particles_new)(
        output_buffer, buf_size_t{ 100 } );
    SIXTRL_ASSERT( pset_dummy != nullptr );

    pset_dummy = ::NS(Particles_new)(
        output_buffer, buf_size_t{ 100 } );
    SIXTRL_ASSERT( pset_dummy != nullptr );
    ( void )pset_dummy;

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        ::NS(Drift)* drift = ::NS(Drift_add)( beam_elements, 0.1 * ii );
        SIXTRL_ASSERT( drift != nullptr );
        ( void )drift;

        be_mon_indices.push_back( ::NS(Buffer_get_num_of_objects)(
            beam_elements ) );

        ::NS(BeamMonitor)* be_mon = ::NS(BeamMonitor_new)( beam_elements );
        SIXTRL_ASSERT( be_mon != nullptr );

        ::NS(BeamMonitor_set_out_address)( be_mon, ::NS(buffer_addr_t){ 0 } );

        out_buffer_indices.push_back( ::NS(Buffer_get_num_of_objects)(
            output_buffer ) );

        particle_set_t* out_pset = ::NS(Particles_new)(
            output_buffer, buf_size_t{ 1 } );

        SIXTRL_ASSERT( out_pset != nullptr );
        ( void )out_pset;
    }

    SIXTRL_ASSERT( out_buffer_indices.size() == be_mon_indices.size() );

    auto be_mon_idx_it  = be_mon_indices.cbegin();
    auto be_mon_idx_end = be_mon_indices.cend();
    auto out_idx_it     = out_buffer_indices.cbegin();

    for( ; be_mon_idx_it != be_mon_idx_end ; ++be_mon_idx_it, ++out_idx_it )
    {
        buf_size_t const be_mon_idx = *be_mon_idx_it;
        buf_size_t const out_pset_idx = *out_idx_it;

        assign_item_t* assign_item = nullptr;

        if( ( be_mon_idx % buf_size_t{ 2 } ) == buf_size_t{ 0 } )
        {
            assign_item = ::NS(AssignAddressItem_new)( map_buffer );
            ASSERT_TRUE( assign_item != nullptr );

            ::NS(AssignAddressItem_set_dest_buffer_id)(
                assign_item, ::NS(ARCH_BEAM_ELEMENTS_BUFFER_ID) );

            ::NS(AssignAddressItem_set_dest_elem_type_id)(
                assign_item, ::NS(OBJECT_TYPE_BEAM_MONITOR) );

            ::NS(AssignAddressItem_set_dest_elem_index)(
                assign_item, be_mon_idx );

            ::NS(AssignAddressItem_set_dest_pointer_offset)(
                assign_item, offsetof( NS(BeamMonitor), out_address ) );

            ::NS(AssignAddressItem_set_src_buffer_id)(
                assign_item, ::NS(ARCH_OUTPUT_BUFFER_ID) );

            ::NS(AssignAddressItem_set_src_elem_type_id)(
                assign_item, ::NS(OBJECT_TYPE_PARTICLE) );

            ::NS(AssignAddressItem_set_src_elem_index)(
                assign_item, out_pset_idx );

            ::NS(AssignAddressItem_set_src_pointer_offset)(
                assign_item, ::NS(buffer_addr_t){ 0 } );
        }
        else
        {
            assign_item = ::NS(AssignAddressItem_add)( map_buffer,
                ::NS(OBJECT_TYPE_BEAM_MONITOR),
                ::NS(ARCH_BEAM_ELEMENTS_BUFFER_ID), be_mon_idx,
                offsetof( NS(BeamMonitor), out_address ),
                ::NS(OBJECT_TYPE_PARTICLE), ::NS(ARCH_OUTPUT_BUFFER_ID),
                out_pset_idx, ::NS(buffer_addr_t){ 0 } );

            ASSERT_TRUE( assign_item != nullptr );
        }

        ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_type_id)( assign_item ) ==
                     ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        ASSERT_TRUE( ::NS(AssignAddressItem_dest_buffer_id)( assign_item ) ==
                     ::NS(ARCH_BEAM_ELEMENTS_BUFFER_ID) );

        ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_index)(
            assign_item ) == be_mon_idx );

        ASSERT_TRUE( ::NS(AssignAddressItem_dest_pointer_offset)(
            assign_item ) == offsetof( ::NS(BeamMonitor), out_address ) );

        ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_type_id)(
            assign_item ) == ::NS(OBJECT_TYPE_PARTICLE) );

        ASSERT_TRUE( ::NS(AssignAddressItem_src_buffer_id)(
            assign_item ) == ::NS(ARCH_OUTPUT_BUFFER_ID) );

        ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_index)(
            assign_item ) == out_pset_idx );

        ASSERT_TRUE( ::NS(AssignAddressItem_dest_pointer_offset)(
            assign_item ) == ::NS(buffer_addr_t){ 0 } );
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)(
        map_buffer ) == NUM_BEAM_MONITORS );

    /* ********************************************************************* */

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        assign_item_t const* assign_item =
            ::NS(AssignAddressItem_buffer_get_const_item)( map_buffer, ii );

        ASSERT_TRUE( assign_item != nullptr );

        be_monitor_t const* be_mon = ::NS(BeamElements_buffer_get_beam_monitor)(
            beam_elements, ::NS(AssignAddressItem_dest_elem_index)(
                assign_item ) );

        SIXTRL_ASSERT( be_mon != nullptr );

        SIXTRL_ASSERT( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            ::NS(buffer_addr_t){ 0 } );

        ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
            NS(AssignAddressItem_assign_fixed_addr)( assign_item,
                beam_elements, static_cast< ::NS(buffer_addr_t) >( ii ) ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            static_cast< ::NS(buffer_addr_t) >( ii ) );

        ASSERT_TRUE( ::NS(AssignAddressItem_remap_assignment)( assign_item,
            beam_elements, ::NS(buffer_addr_t){ 100 } ) ==
                ::NS(ARCH_STATUS_SUCCESS) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            static_cast< ::NS(buffer_addr_t) >( ii + 100 ) );

        ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
            ::NS(AssignAddressItem_remap_assignment)( assign_item,
                beam_elements, -( static_cast< ::NS(buffer_addr_diff_t) >(
                    ii + 100 ) ) ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            ::NS(buffer_addr_t){ 0 } );
    }

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        assign_item_t const* assign_item =
            ::NS(AssignAddressItem_buffer_get_const_item)( map_buffer, ii );

        particle_set_t const* pset = ::NS(Particles_buffer_get_const_particles)(
            output_buffer, out_buffer_indices[ ii ] );

        be_monitor_t const* be_mon =
            ::NS(BeamElements_buffer_get_const_beam_monitor)( beam_elements,
                be_mon_indices[ ii ] );

        SIXTRL_ASSERT( assign_item != nullptr );
        SIXTRL_ASSERT( pset != nullptr );
        SIXTRL_ASSERT( be_mon != nullptr );

        ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
            ::NS(AssignAddressItem_perform_assignment)( assign_item,
                beam_elements, output_buffer ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            static_cast< ::NS(buffer_addr_t) >( reinterpret_cast< uintptr_t >(
                pset ) ) );

        ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
            ::NS(AssignAddressItem_assign_fixed_addr)( assign_item,
                beam_elements, ::NS(buffer_addr_t){ 0 } ) );
    }

    /* ********************************************************************* */

    ::NS(Buffer_delete)( map_buffer );
    ::NS(Buffer_delete)( beam_elements );
    ::NS(Buffer_delete)( output_buffer );
}

/* end: tests/sixtracklib/common/buffer/test_assign_address_item_c99.cpp */
