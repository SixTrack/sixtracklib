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
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"

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
    out_buffer_indices.clear();

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

        ASSERT_TRUE( ::NS(AssignAddressItem_src_pointer_offset)(
            assign_item ) == ::NS(buffer_addr_t){ 0 } );
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)(
        map_buffer ) == NUM_BEAM_MONITORS );

    /* ********************************************************************* */

    buf_size_t const dest_slot_size =
        ::NS(Buffer_get_slot_size)( beam_elements );

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
                beam_elements, static_cast< ::NS(buffer_addr_t) >(
                    ii * dest_slot_size ) ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            static_cast< ::NS(buffer_addr_t) >( ii * dest_slot_size ) );

        ASSERT_TRUE( ::NS(AssignAddressItem_remap_assignment)( assign_item,
            beam_elements, ::NS(buffer_addr_t){ 192 } ) ==
                ::NS(ARCH_STATUS_SUCCESS) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_mon ) ==
            static_cast< ::NS(buffer_addr_t) >( ii * dest_slot_size + 192 ) );

        ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
            ::NS(AssignAddressItem_remap_assignment)( assign_item,
                beam_elements, -( static_cast< ::NS(buffer_addr_diff_t) >(
                    ii * dest_slot_size + 192 ) ) ) );

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

TEST( C99_Common_Buffer_AssignAddressItemTests, ElemByElemConfigTest )
{
    using elem_config_t     = ::NS(ElemByElemConfig);
    using buffer_t          = ::NS(Buffer);
    using buf_size_t        = ::NS(buffer_size_t);
    using assign_item_t     = ::NS(AssignAddressItem);
    using particle_set_t    = ::NS(Particles);
    using addr_t            = ::NS(buffer_addr_t);

    buffer_t* map_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* elem_by_elem_config_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* output_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    elem_config_t config_on_stack;
    ::NS(ElemByElemConfig_preset)( &config_on_stack );
    ::NS(ElemByElemConfig_set_output_store_address)(
        &config_on_stack, addr_t{ 0 } );

    elem_config_t* config_in_buffer =
        ::NS(ElemByElemConfig_new)( elem_by_elem_config_buffer );

    SIXTRL_ASSERT( config_in_buffer != nullptr );
    ::NS(ElemByElemConfig_set_output_store_address)(
        config_in_buffer, addr_t{ 0 } );

    particle_set_t* dummy_pset = ::NS(Particles_new)(
        output_buffer, buf_size_t{ 1 } );
    SIXTRL_ASSERT( dummy_pset != nullptr );
    ( void )dummy_pset;

    particle_set_t* output_pset = ::NS(Particles_new)(
        output_buffer, buf_size_t{ 1 } );
    SIXTRL_ASSERT( output_pset != nullptr );

    /* --------------------------------------------------------------------- */
    /* Create assignment item for elem-by-elem-config on stack */

    assign_item_t* assign_item_stack = ::NS(AssignAddressItem_add)( map_buffer,
        ::NS(OBJECT_TYPE_NONE), ::NS(ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID),
        buf_size_t{ 0 }, offsetof( ::NS(ElemByElemConfig), out_store_addr ),
        ::NS(OBJECT_TYPE_PARTICLE), ::NS(ARCH_OUTPUT_BUFFER_ID),
        buf_size_t{ 1 }, buf_size_t{ 0 } );

    ASSERT_TRUE( assign_item_stack != nullptr );

    /* --------------------------------------------------------------------- */
    /* Create assignment item for elem-by-elem-config in buffer */

    assign_item_t* assign_item_buffer =
        ::NS(AssignAddressItem_new)( map_buffer );

    ASSERT_TRUE( assign_item_buffer != nullptr );

    ::NS(AssignAddressItem_set_dest_elem_type_id)(
        assign_item_buffer, ::NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) );

    ::NS(AssignAddressItem_set_dest_buffer_id)(
        assign_item_buffer, ::NS(ARCH_MIN_USER_DEFINED_BUFFER_ID) );

    ::NS(AssignAddressItem_set_dest_elem_index)(
        assign_item_buffer, buf_size_t{ 0 } );

    ::NS(AssignAddressItem_set_dest_pointer_offset)(
        assign_item_buffer, offsetof( ::NS(ElemByElemConfig), out_store_addr ) );

    ::NS(AssignAddressItem_set_src_elem_type_id)(
        assign_item_buffer, ::NS(OBJECT_TYPE_PARTICLE) );

    ::NS(AssignAddressItem_set_src_buffer_id)(
        assign_item_buffer, ::NS(ARCH_OUTPUT_BUFFER_ID) );

    ::NS(AssignAddressItem_set_src_elem_index)(
        assign_item_buffer, buf_size_t{ 1 } );

    ::NS(AssignAddressItem_set_src_pointer_offset)(
        assign_item_buffer, buf_size_t{ 0 } );

    /* --------------------------------------------------------------------- */
    /* Perform assignments and remappings for elem-by-elem config on stack */

    assign_item_stack = ::NS(AssignAddressItem_buffer_get_item)( map_buffer, 0 );
    ASSERT_TRUE( assign_item_stack != nullptr );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_buffer_id)(
        assign_item_stack ) == ::NS(ASSIGN_ADDRESS_ITEM_NO_BUFFER_ID) );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_type_id)(
        assign_item_stack ) == ::NS(OBJECT_TYPE_NONE) );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_index)(
        assign_item_stack ) == buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_pointer_offset)( assign_item_stack
        ) == offsetof( ::NS(ElemByElemConfig), out_store_addr ) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_buffer_id)(
        assign_item_stack ) == ::NS(ARCH_OUTPUT_BUFFER_ID) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_type_id)(
        assign_item_stack ) == ::NS(OBJECT_TYPE_PARTICLE) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_index)(
        assign_item_stack ) == buf_size_t{ 1 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_pointer_offset)(
        assign_item_stack ) == buf_size_t{ 0 } );

    unsigned char* ptr_stack_begin = reinterpret_cast< unsigned char* >(
        reinterpret_cast< uintptr_t >( &config_on_stack ) );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        assign_item_stack, ptr_stack_begin, buf_size_t{ 0 }, addr_t{ 42 } ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        &config_on_stack ) == addr_t{ 42 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_remap_assignment)(
        assign_item_stack, ptr_stack_begin, buf_size_t{ 0 },
            ::NS(buffer_addr_diff_t){ 214 } ) == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        &config_on_stack ) == addr_t{ 256 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_remap_assignment)(
        assign_item_stack, ptr_stack_begin, buf_size_t{ 0 },
            ::NS(buffer_addr_diff_t){ -214 } ) == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        &config_on_stack ) == addr_t{ 42 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        assign_item_stack, ptr_stack_begin, buf_size_t{ 0 }, addr_t{ 0 } ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        &config_on_stack ) == addr_t{ 0 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_perform_assignment)(
        assign_item_stack, ptr_stack_begin, buf_size_t{ 0 },
            ::NS(Buffer_get_data_begin)( output_buffer ),
                ::NS(Buffer_get_slot_size)( output_buffer ) ) ==
                    ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        &config_on_stack ) == static_cast< addr_t >( reinterpret_cast<
            uintptr_t >( output_pset ) ) );

    /* --------------------------------------------------------------------- */
    /* Perform assignments and remappings for elem-by-elem config in buffer  */

    assign_item_buffer = ::NS(AssignAddressItem_buffer_get_item)(
        map_buffer, 1 );

    ASSERT_TRUE( assign_item_buffer != nullptr );
    ASSERT_TRUE( ::NS(AssignAddressItem_dest_buffer_id)(
        assign_item_buffer ) == ::NS(ARCH_MIN_USER_DEFINED_BUFFER_ID) );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_type_id)(
        assign_item_buffer ) == ::NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_elem_index)(
        assign_item_buffer ) == buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_dest_pointer_offset)( assign_item_buffer
        ) == offsetof( ::NS(ElemByElemConfig), out_store_addr ) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_buffer_id)(
        assign_item_buffer ) == ::NS(ARCH_OUTPUT_BUFFER_ID) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_type_id)(
        assign_item_buffer ) == ::NS(OBJECT_TYPE_PARTICLE) );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_elem_index)(
        assign_item_buffer ) == buf_size_t{ 1 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_src_pointer_offset)(
        assign_item_buffer ) == buf_size_t{ 0 } );

    buf_size_t const dest_slot_size =
        ::NS(Buffer_get_slot_size)( elem_by_elem_config_buffer );

    unsigned char* ptr_buffer_begin =
        ::NS(Buffer_get_data_begin)( elem_by_elem_config_buffer );

    SIXTRL_ASSERT( dest_slot_size > buf_size_t{ 0 } );
    SIXTRL_ASSERT( ptr_buffer_begin != nullptr );

    addr_t const fixed_addr = static_cast< addr_t >( dest_slot_size );
    addr_t const remapped_fixed_addr = static_cast< addr_t >(
        10 * dest_slot_size );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        assign_item_buffer, ptr_buffer_begin, dest_slot_size,
            static_cast< addr_t >( dest_slot_size ) ) ==
                ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        config_in_buffer ) == static_cast< addr_t >( dest_slot_size ) );

    ::NS(buffer_addr_diff_t) dist = static_cast< addr_t >(
        9 * dest_slot_size );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_remap_assignment)(
        assign_item_buffer, ptr_buffer_begin, dest_slot_size, dist ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        config_in_buffer ) == remapped_fixed_addr );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_remap_assignment)(
        assign_item_buffer, ptr_buffer_begin, dest_slot_size, -dist ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        config_in_buffer ) == fixed_addr );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_assign_fixed_addr)(
        assign_item_buffer, ptr_buffer_begin, dest_slot_size, addr_t{ 0 } ) ==
            ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        config_in_buffer ) == addr_t{ 0 } );

    ASSERT_TRUE( ::NS(AssignAddressItem_managed_buffer_perform_assignment)(
        assign_item_buffer, ptr_buffer_begin, dest_slot_size,
            ::NS(Buffer_get_data_begin)( output_buffer ),
                ::NS(Buffer_get_slot_size)( output_buffer ) ) ==
                    ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_output_store_address)(
        config_in_buffer ) == static_cast< addr_t >( reinterpret_cast<
            uintptr_t >( output_pset ) ) );

    /* ********************************************************************* */

    ::NS(Buffer_delete)( map_buffer );
    ::NS(Buffer_delete)( elem_by_elem_config_buffer );
    ::NS(Buffer_delete)( output_buffer );
}

/* end: tests/sixtracklib/common/buffer/test_assign_address_item_c99.cpp */
