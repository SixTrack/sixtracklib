#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/managed_buffer.h"

/* ========================================================================= */
/* ====  Test flat-memory based minimal buffer implementation                */

TEST( C99_CommonManagedBuffer, InitAndReserve )
{
    using size_t = ::st_buffer_size_t;

    std::vector< unsigned char > data_buffer( size_t{ 4096 }, uint8_t{ 0 } );
    auto begin = data_buffer.data();

    constexpr size_t const SLOTS_ID     = size_t{ 3 };
    constexpr size_t const OBJECTS_ID   = size_t{ 4 };
    constexpr size_t const DATAPTRS_ID  = size_t{ 5 };
    constexpr size_t const GARBAGE_ID   = size_t{ 6 };
    constexpr size_t const slot_size    = st_BUFFER_DEFAULT_SLOT_SIZE;

    size_t max_num_objects              = size_t{ 0 };
    size_t max_num_slots                = size_t{ 0 };
    size_t max_num_dataptrs             = size_t{ 0 };
    size_t max_num_garbage_range        = size_t{ 0 };

    size_t predicted_size = st_ManagedBuffer_calculate_buffer_length( begin,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, slot_size );

    ASSERT_TRUE( predicted_size > size_t{ 0 } );

    size_t current_buffer_size  = size_t{ 0 };

    int success = st_ManagedBuffer_init( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( predicted_size == current_buffer_size );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, SLOTS_ID, slot_size ) == max_num_slots );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, OBJECTS_ID, slot_size ) == max_num_objects );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, DATAPTRS_ID, slot_size ) == max_num_dataptrs );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, GARBAGE_ID, slot_size ) == max_num_garbage_range );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, SLOTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, OBJECTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, GARBAGE_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_buffer_length( begin, slot_size ) ==
        current_buffer_size );

    /* -------------------------------------------------------------------- */

    max_num_objects       = size_t{ 1 };
    max_num_slots         = size_t{ 8 };
    max_num_dataptrs      = size_t{ 0 };
    max_num_garbage_range = size_t{ 0 };

    predicted_size = st_ManagedBuffer_calculate_buffer_length( begin,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, slot_size );

    ASSERT_TRUE( predicted_size > current_buffer_size );
    ASSERT_TRUE( predicted_size < data_buffer.size()  );

    ASSERT_TRUE( st_ManagedBuffer_can_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size ) );

    success = st_ManagedBuffer_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, SLOTS_ID, slot_size ) == max_num_slots );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, OBJECTS_ID, slot_size ) == max_num_objects );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, DATAPTRS_ID, slot_size ) == max_num_dataptrs );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, GARBAGE_ID, slot_size ) == max_num_garbage_range );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, SLOTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, OBJECTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, GARBAGE_ID, slot_size ) == size_t{ 0 } );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    using address_t     = ::st_buffer_addr_t;
    using ptr_to_addr_t = address_t*;

    std::vector< address_t >  cmp_slot_values( max_num_slots, address_t{ 0 } );

    address_t cnt = address_t{ 0 };

    ptr_to_addr_t slot_it = reinterpret_cast< ptr_to_addr_t >(
        st_ManagedBuffer_get_ptr_to_section_data(
            begin, SLOTS_ID, slot_size ) );

    ptr_to_addr_t slot_end = slot_it;
    std::advance( slot_end, max_num_slots );

    auto cmp_slot_it = cmp_slot_values.begin();

    for( ; slot_it != slot_end ; ++slot_it, ++cnt, ++cmp_slot_it )
    {
        *slot_it     = cnt;
        *cmp_slot_it = cnt;
    }

    st_ManagedBuffer_set_section_num_entities(
        begin, SLOTS_ID, max_num_slots, slot_size );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    using type_id_t     = ::st_object_type_id_t;
    using object_t      = ::st_Object;
    using ptr_to_obj_t  = object_t*;

    std::vector< object_t > cmp_object_values( max_num_objects, object_t{} );

    ptr_to_obj_t obj_it  = reinterpret_cast< ptr_to_obj_t >(
        st_ManagedBuffer_get_ptr_to_section_data(
            begin, OBJECTS_ID, slot_size ) );

    ptr_to_obj_t  obj_end = obj_it;
    std::advance( obj_end, max_num_objects );

    auto cmp_obj_it = cmp_object_values.begin();

    type_id_t  type_id = type_id_t{ 1 };
    size_t size = sizeof( address_t );

    for( ; obj_it != obj_end ; ++obj_it, ++type_id, ++cmp_obj_it )
    {
        st_Object_preset( obj_it );
        st_Object_set_type_id( obj_it, type_id );
        st_Object_set_size( obj_it, size );

        *cmp_obj_it = *obj_it;
    }

    st_ManagedBuffer_set_section_num_entities(
        begin, OBJECTS_ID, max_num_objects, slot_size );

    /* -------------------------------------------------------------------- */

    size_t cur_num_objects  = st_ManagedBuffer_get_section_num_entities(
        begin, OBJECTS_ID, slot_size );

    size_t cur_num_slots    = st_ManagedBuffer_get_section_num_entities(
        begin, SLOTS_ID, slot_size );

    size_t cur_num_dataptrs = st_ManagedBuffer_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size );

    size_t cur_num_garbage_range = st_ManagedBuffer_get_section_num_entities(
        begin, GARBAGE_ID, slot_size );

    max_num_objects       += size_t{ 2 };
    max_num_slots         += size_t{ 8 };
    max_num_dataptrs      += size_t{ 4 };
    max_num_garbage_range += size_t{ 1 };

    predicted_size = st_ManagedBuffer_calculate_buffer_length( begin,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, slot_size );

    ASSERT_TRUE( predicted_size > current_buffer_size );
    ASSERT_TRUE( predicted_size < data_buffer.size()  );

    ASSERT_TRUE( st_ManagedBuffer_can_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size ) );

    success = st_ManagedBuffer_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, SLOTS_ID, slot_size ) == max_num_slots );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, OBJECTS_ID, slot_size ) == max_num_objects );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, DATAPTRS_ID, slot_size ) == max_num_dataptrs );

    ASSERT_TRUE( st_ManagedBuffer_get_section_max_num_entities(
        begin, GARBAGE_ID, slot_size ) == max_num_garbage_range );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, SLOTS_ID, slot_size ) == cur_num_slots );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, OBJECTS_ID, slot_size ) == cur_num_objects );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size ) == cur_num_dataptrs );

    ASSERT_TRUE( st_ManagedBuffer_get_section_num_entities(
        begin, GARBAGE_ID, slot_size ) == cur_num_garbage_range );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    slot_it = reinterpret_cast< ptr_to_addr_t >(
        st_ManagedBuffer_get_ptr_to_section_data(
            begin, SLOTS_ID, slot_size ) );

    slot_end = reinterpret_cast< ptr_to_addr_t >(
        st_ManagedBuffer_get_ptr_to_section_end(
            begin, SLOTS_ID, slot_size ) );

    ASSERT_TRUE( std::distance( slot_it, slot_end ) ==
        static_cast< std::ptrdiff_t >(
            st_ManagedBuffer_get_section_num_entities(
                begin, SLOTS_ID, slot_size ) ) );

    cmp_slot_it = cmp_slot_values.begin();

    for( ; slot_it != slot_end ; ++slot_it, ++cmp_slot_it )
    {
        ASSERT_TRUE( *slot_it == *cmp_slot_it );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_it  = reinterpret_cast< ptr_to_obj_t >(
        st_ManagedBuffer_get_ptr_to_section_data(
            begin, OBJECTS_ID, slot_size ) );

    obj_end = reinterpret_cast< ptr_to_obj_t >(
        st_ManagedBuffer_get_ptr_to_section_end(
            begin, OBJECTS_ID, slot_size ) );

    ASSERT_TRUE( std::distance( obj_it, obj_end ) ==
        static_cast< std::ptrdiff_t >(
            st_ManagedBuffer_get_section_num_entities(
                begin, OBJECTS_ID, slot_size ) ) );

    cmp_obj_it = cmp_object_values.begin();

    for( ; obj_it != obj_end ; ++obj_it, ++type_id, ++cmp_obj_it )
    {
        ASSERT_TRUE( st_Object_get_type_id( obj_it ) ==
                     st_Object_get_type_id( &( *cmp_obj_it ) ) );

        ASSERT_TRUE( st_Object_get_size( obj_it ) ==
                     st_Object_get_size( &( *cmp_obj_it ) ) );

        ASSERT_TRUE( st_Object_get_begin_addr( obj_it ) ==
                     st_Object_get_begin_addr( &( *cmp_obj_it ) ) );
    }
}

/* ------------------------------------------------------------------------- */
/* end: tests/sixtracklib/common/test_managed_buffer_c99.cpp */
