#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/buffer_mem.h"

/* ========================================================================= */
/* ====  Test flat-memory based minimal buffer implementation                */

TEST( C99_CommonBufferMem, InitAndReserve )
{
    using size_t = ::st_buffer_size_t;

    std::vector< unsigned char > data_buffer( size_t{ 4096 }, uint8_t{ 0 } );
    auto begin = data_buffer.data();

    constexpr size_t const SLOTS_ID     = size_t{ 3 };
    constexpr size_t const OBJECTS_ID   = size_t{ 4 };
    constexpr size_t const DATAPTRS_ID  = size_t{ 5 };
    constexpr size_t const GARBAGE_ID   = size_t{ 6 };

    size_t const slot_size       = st_BUFFER_DEFAULT_SLOT_SIZE;

    size_t max_num_objects       = size_t{ 0 };
    size_t max_num_slots         = size_t{ 0 };
    size_t max_num_dataptrs      = size_t{ 0 };
    size_t max_num_garbage_range = size_t{ 0 };

    size_t predicted_size = st_BufferMem_calculate_buffer_length( begin,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, slot_size );

    ASSERT_TRUE( predicted_size > size_t{ 0 } );

    size_t current_buffer_size   = size_t{ 0 };

    int success = st_BufferMem_init( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( predicted_size == current_buffer_size );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, SLOTS_ID, slot_size ) == max_num_slots );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, OBJECTS_ID, slot_size ) == max_num_objects );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, DATAPTRS_ID, slot_size ) == max_num_dataptrs );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, GARBAGE_ID, slot_size ) == max_num_garbage_range );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, SLOTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, OBJECTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, GARBAGE_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_buffer_length( begin, slot_size ) ==
        current_buffer_size );

    /* -------------------------------------------------------------------- */

    max_num_objects       = size_t{ 1 };
    max_num_slots         = size_t{ 8 };
    max_num_dataptrs      = size_t{ 0 };
    max_num_garbage_range = size_t{ 0 };

    predicted_size = st_BufferMem_calculate_buffer_length( begin,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, slot_size );

    ASSERT_TRUE( predicted_size > current_buffer_size );
    ASSERT_TRUE( predicted_size < data_buffer.size()  );

    ASSERT_TRUE( st_BufferMem_can_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size ) );

    success = st_BufferMem_reserve( begin, &current_buffer_size,
        max_num_objects, max_num_slots, max_num_dataptrs,
            max_num_garbage_range, data_buffer.size(), slot_size );

    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, SLOTS_ID, slot_size ) == max_num_slots );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, OBJECTS_ID, slot_size ) == max_num_objects );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, DATAPTRS_ID, slot_size ) == max_num_dataptrs );

    ASSERT_TRUE( st_BufferMem_get_section_max_num_entities(
        begin, GARBAGE_ID, slot_size ) == max_num_garbage_range );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, SLOTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, OBJECTS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, DATAPTRS_ID, slot_size ) == size_t{ 0 } );

    ASSERT_TRUE( st_BufferMem_get_section_num_entities(
        begin, GARBAGE_ID, slot_size ) == size_t{ 0 } );


}

/* ------------------------------------------------------------------------- */
/* end: tests/sixtracklib/common/test_buffer_mem.cpp */
