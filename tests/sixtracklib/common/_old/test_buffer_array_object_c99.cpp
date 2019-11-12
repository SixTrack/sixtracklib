#include "sixtracklib/common/buffer/buffer_array_object.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/testlib.h"

TEST( C99_CommonBufferArrayObjectTests, BasicUsage )
{
    using buffer_t   = ::NS(Buffer)*;
    using buf_size_t = ::NS(buffer_size_t);
    using addr_t     = ::NS(buffer_addr_t);
    using arrobj_t   = ::NS(BufferArrayObj);
    using object_t   = ::NS(Object);
    using type_id_t  = ::NS(object_type_id_t);

    buffer_t buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    buf_size_t const arr1_max_num_elements = buf_size_t{ 10 };
    buf_size_t const arr1_capacity = buf_size_t{ 1 << 12 };

    arrobj_t* arr1 = ::NS(BufferArrayObj_new)(
        buffer, arr1_max_num_elements, arr1_capacity, type_id_t{ 1 } );

    ASSERT_TRUE( arr1 != nullptr );
    ASSERT_TRUE( ::NS(BufferArrayObj_get_capacity)( arr1 ) == arr1_capacity );
    ASSERT_TRUE( ::NS(BufferArrayObj_get_max_num_elements)( arr1 ) ==
                 arr1_max_num_elements );

    ASSERT_TRUE( ::NS(BufferArrayObj_get_slot_size)( arr1 ) ==
                 ::NS(BUFFER_DEFAULT_SLOT_SIZE) );

    ASSERT_TRUE( ::NS(BufferArrayObj_get_num_elements)( arr1 ) ==
                 buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(BufferArrayObj_get_length)( arr1 ) == buf_size_t{ 0 } );


    ::NS(Buffer_delete)( buffer );
    buffer = nullptr;
}

/* end: tests/sixtracklib/common/test_buffer_array_object_c99.cpp */
