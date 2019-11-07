#include "sixtracklib/common/buffer/buffer_string_object.h"

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

TEST( C99_CommonBufferStringObjectTests, BasicUsage )
{
    using buffer_t     = ::NS(Buffer)*;
    using buf_size_t   = ::NS(buffer_size_t);
    using addr_t       = ::NS(buffer_addr_t);
    using strobj_t     = ::NS(BufferStringObj)*;
    using object_t     = ::NS(Object);

    buffer_t buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    char const cstr_base[]           = "test 123 test 456 ";
    char const cstr_append1[]        = "test 789 ";
    char const cstr_after_append1[]  = "test 123 test 456 test 789 ";
    char const cstr_append2[]        = "tes";
    char const cstr_after_append2[]  = "test 123 test 456 test 789 tes";
    char const cstr_append_fail[]    = "test";
    char const cstr_too_long[]       = "test 123 test 456 test 789 test";

    buf_size_t const cstr_max_length = std::strlen( cstr_after_append2 );
    buf_size_t const str1_capacity   = cstr_max_length + buf_size_t{ 1 };

    strobj_t str1 = ::NS(BufferStringObj_new)( buffer, cstr_max_length );

    ASSERT_TRUE( str1 != nullptr );
    ASSERT_TRUE( ::NS(BufferStringObj_get_begin_addr)( str1 ) > addr_t{0} );
    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) == buf_size_t{ 0 } );
    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) == str1_capacity );
    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) ==
                 cstr_max_length );

    ASSERT_TRUE( ::NS(BufferStringObj_get_const_string)( str1 ) != nullptr );
    ASSERT_TRUE( std::strlen( ::NS(BufferStringObj_get_const_string)( str1 ) )
                 == buf_size_t{ 0 } );

    object_t const* obj =
        ::NS(Buffer_get_const_object)( buffer, buf_size_t{ 0 } );

    ASSERT_TRUE( obj != nullptr );
    ASSERT_TRUE( ::NS(Object_get_type_id)( obj ) ==
                 ::NS(OBJECT_TYPE_CSTRING) );

    strobj_t str2 = ::NS(BufferStringObj_new_from_cstring)( buffer, cstr_base );

    /* inserting an object into a buffer can invalidate all exisiting
     * references to objects as the buffer may have to re-allocated in order
     * to accomodate the new elements. Thus, we have to reacquire str1 */

    str1 = ::NS(BufferStringObj_get_from_buffer)( buffer, buf_size_t{ 0 } );
    ASSERT_TRUE( str1 != nullptr );

    /* Check whether the invariantes for str1 still hold before moving on to
     * checking the invariants for str2 */

    ASSERT_TRUE( ::NS(BufferStringObj_get_begin_addr)( str1 ) > addr_t{0} );
    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) == buf_size_t{ 0 } );
    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) == str1_capacity );
    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) ==
                 cstr_max_length );

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) <
                 std::strlen( cstr_too_long ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_const_string)( str1 ) != nullptr );
    ASSERT_TRUE( std::strlen( ::NS(BufferStringObj_get_const_string)( str1 ) )
                 == buf_size_t{ 0 } );

    /* Check the invariants of str2 */

    ASSERT_TRUE( str2 != nullptr );
    ASSERT_TRUE( str2 != str1 );

    ASSERT_TRUE( ::NS(BufferStringObj_get_begin_addr)( str2 ) > addr_t{ 0 } );
    ASSERT_TRUE( ::NS(BufferStringObj_get_begin_addr)( str2 ) !=
                 ::NS(BufferStringObj_get_begin_addr)( str1 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str2 ) ==
                 std::strlen( cstr_base ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str2 ) ==
                 ( std::strlen( cstr_base ) + buf_size_t{ 1 } ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str2 ) ==
                 std::strlen( cstr_base ) );

    ASSERT_TRUE( std::strcmp( ::NS(BufferStringObj_get_const_string)( str2 ),
                              cstr_base ) == 0 );

    /* copy the string stored in str2 into str1 and compare stored states */

    ASSERT_TRUE( nullptr != ::NS(BufferStringObj_assign_cstring)(
        str1, cstr_base ) );

    ASSERT_TRUE( std::strcmp( ::NS(BufferStringObj_get_const_string)( str1 ),
                    ::NS(BufferStringObj_get_const_string)( str2 ) ) == 0 );

    /* append a string to str1. str1 and str2 no longer store the same state */

    ASSERT_TRUE( nullptr != ::NS(BufferStringObj_append_cstring)(
        str1, cstr_append1 ) );

    ASSERT_TRUE( std::strcmp( ::NS(BufferStringObj_get_const_string)( str1 ),
                    cstr_after_append1 ) == 0 );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) ==
                 std::strlen( cstr_after_append1 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) == str1_capacity );

    /* attempt to append another snipplet of a string -> this should fail
     * as the resulting string would exceed the NS(BufferStrinObj)'s
     * capacity */

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) <
                 std::strlen( cstr_too_long ) );

    ASSERT_TRUE( nullptr == ::NS(BufferStringObj_append_cstring)(
        str1, cstr_append_fail ) );

    /* But adding a slightly shorter string should work: */

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) ==
                 std::strlen( cstr_after_append2 ) );

    ASSERT_TRUE( nullptr != ::NS(BufferStringObj_append_cstring)(
        str1, cstr_append2 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) ==
                 ::NS(BufferStringObj_get_length)( str1 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_max_length)( str1 ) ==
                 std::strlen( cstr_after_append2 ) );

    /* Compare results to the expected stored state */

    ASSERT_TRUE( std::strcmp( ::NS(BufferStringObj_get_const_string)( str1 ),
                    cstr_after_append2 ) == 0 );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) ==
                 std::strlen( cstr_after_append2 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) ==
                 ::NS(BufferStringObj_get_max_length)( str1 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) == str1_capacity );
    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) >
                 ::NS(BufferStringObj_get_max_length)( str1 ) );

    /* Appending any string to str2 does not work, however as str2 is already
     * at the maximum length */

    ASSERT_TRUE( nullptr == ::NS(BufferStringObj_append_cstring)(
        str2, cstr_append2 ) );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str2 ) ==
                 ::NS(BufferStringObj_get_max_length)( str2 ) );

    /* Verify that clearing str1 results in an empty string stored */

    ::NS(BufferStringObj_clear)( str1 );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) == buf_size_t{ 0 } );
    ASSERT_TRUE( ::NS(BufferStringObj_get_capacity)( str1 ) == str1_capacity );
    ASSERT_TRUE( std::strlen( ::NS(BufferStringObj_get_const_string)( str1 ) )
                 == buf_size_t{ 0 } );

    /* You can use the "regular" string functions with a NS(BufferStringObj);
     * but you have to manually sync its internal state whenevery the
     * length of the string has been altered */

    std::strncpy( ::NS(BufferStringObj_get_string)( str1 ),
                  ::NS(BufferStringObj_get_const_string)( str2 ),
                  ::NS(BufferStringObj_get_max_length)( str1 ) );

    ::NS(BufferStringObj_sync_length)( str1 );

    ASSERT_TRUE( ::NS(BufferStringObj_get_length)( str1 ) ==
                 ::NS(BufferStringObj_get_length)( str2 ) );

    ASSERT_TRUE( 0 == std::strcmp(
        ::NS(BufferStringObj_get_const_string)( str1 ),
        ::NS(BufferStringObj_get_const_string)( str2 ) ) );

    str1 = nullptr;
    str2 = nullptr;

    ::NS(Buffer_delete)( buffer );
    buffer = nullptr;
}

/* end: tests/sixtracklib/common/test_buffer_string_object_c99.cpp */
