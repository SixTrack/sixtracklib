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
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer/buffer_string_object.hpp"
#include "sixtracklib/testlib.h"

TEST( CXX_CommonBufferStringObjectTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using strobj_t     = st::BufferStringObj;
    using buffer_t     = st::Buffer;
    using object_t     = buffer_t::object_t;
    using buf_size_t   = buffer_t::size_type;
    using addr_t       = buffer_t::address_t;

    buffer_t buffer;

    char const cstr_base[]            = "test 123 test 456 ";
    char const cstr_append1[]         = "test 789 ";
    char const cstr_after_append1[]   = "test 123 test 456 test 789 ";
    char const cstr_append2[]         = "tes";
    char const cstr_after_append2[]   = "test 123 test 456 test 789 tes";
    char const cstr_append_fail[]     = "test";
    char const cstr_too_long[]        = "test 123 test 456 test 789 test";

    buf_size_t const cstr_base_length = std::strlen( cstr_base );
    buf_size_t const cstr_append1_len = std::strlen( cstr_after_append1 );
    buf_size_t const cstr_max_length  = std::strlen( cstr_after_append2 );
    buf_size_t const str1_capacity    = cstr_max_length + buf_size_t{ 1 };

    strobj_t* str1 = buffer.createNew< strobj_t >( cstr_max_length );

    ASSERT_TRUE( str1 != nullptr );
    ASSERT_TRUE( str1->getBeginAddress() > addr_t{0} );
    ASSERT_TRUE( str1->getLength() == buf_size_t{ 0 } );
    ASSERT_TRUE( str1->getCapacity() == str1_capacity );
    ASSERT_TRUE( str1->getMaxLength() == cstr_max_length );

    ASSERT_TRUE( str1->getCString() != nullptr );
    ASSERT_TRUE( std::strlen( str1->getCString() ) == buf_size_t{ 0 } );

    object_t const* obj = buffer[ 0 ];
    ASSERT_TRUE( obj != nullptr );
    ASSERT_TRUE( ::NS(Object_get_type_id)( obj ) == ::NS(OBJECT_TYPE_CSTRING));

    strobj_t* str2 = buffer.add< strobj_t >( cstr_base );

    /* inserting an object into a buffer can invalidate all exisiting
     * references to objects as the buffer may have to re-allocated in order
     * to accomodate the new elements. Thus, we have to reacquire str1 */
    str1 = strobj_t::FromBuffer( buffer, buf_size_t{ 0 } );
    ASSERT_TRUE( str1 != nullptr );

    /* Check whether the invariantes for str1 still hold before moving on to
     * checking the invariants for str2 */

    ASSERT_TRUE( str1 != nullptr );
    ASSERT_TRUE( str1->getBeginAddress() > addr_t{0} );
    ASSERT_TRUE( str1->getLength() == buf_size_t{ 0 } );
    ASSERT_TRUE( str1->getCapacity() == str1_capacity );
    ASSERT_TRUE( str1->getMaxLength() == cstr_max_length );
    ASSERT_TRUE( str1->getMaxLength() < std::strlen( cstr_too_long ) );
    ASSERT_TRUE( str1->getCString() != nullptr );
    ASSERT_TRUE( std::strlen( str1->getCString() ) == buf_size_t{ 0 } );

    /* Check the invariants of str2 */

    ASSERT_TRUE( str2 != nullptr );
    ASSERT_TRUE( str2 != str1 );
    ASSERT_TRUE( str2->getBeginAddress() > addr_t{ 0 } );
    ASSERT_TRUE( str2->getBeginAddress() != str1->getBeginAddress() );
    ASSERT_TRUE( str2->getLength() == std::strlen( cstr_base ) );
    ASSERT_TRUE( str2->getCapacity() == cstr_base_length + buf_size_t{ 1 } );
    ASSERT_TRUE( str2->getMaxLength() == std::strlen( cstr_base ) );
    ASSERT_TRUE( std::strcmp( str2->getCString(), cstr_base ) == 0 );

    /* copy the string stored in str2 into str1 and compare stored states */

    ASSERT_TRUE( str1->assign( cstr_base ) != nullptr );
    ASSERT_TRUE( std::strcmp( str1->getCString(), str2->getCString() ) == 0 );

    /* append a string to str1. str1 and str2 no longer store the same state */

    ASSERT_TRUE( str1->append( cstr_append1 ) != nullptr );
    ASSERT_TRUE( str1->getLength() == cstr_append1_len );
    ASSERT_TRUE( std::strcmp( str1->getCString(), cstr_after_append1 ) == 0 );

    /* Repeat to test alternative APIs from the same overload set */

    str1->clear();
    ASSERT_TRUE( ( *str1 += cstr_base ) != nullptr );
    ASSERT_TRUE( std::strcmp( str1->getCString(), str2->getCString() ) == 0 );
    ASSERT_TRUE( str1->append( cstr_append1 ) );
    ASSERT_TRUE( str1->getLength() == cstr_append1_len );
    ASSERT_TRUE( std::strcmp( str1->getCString(), cstr_after_append1 ) == 0 );

    ASSERT_TRUE( nullptr != str1->assign( cstr_base,
            cstr_base + cstr_base_length ) );
    ASSERT_TRUE( std::strcmp( str1->getCString(), str2->getCString() ) == 0 );
    ASSERT_TRUE( nullptr != str1->append( std::string( cstr_append1 ) ) );
    ASSERT_TRUE( str1->getLength() == cstr_append1_len );
    ASSERT_TRUE( std::strcmp( str1->getCString(), cstr_after_append1 ) == 0 );

    ASSERT_TRUE( str1->getCapacity() == str1_capacity );

    /* attempt to append another snipplet of a string -> this should fail
     * as the resulting string would exceed the NS(BufferStrinObj)'s
     * capacity */

    ASSERT_TRUE( str1->getMaxLength() < std::strlen( cstr_too_long ) );
    ASSERT_TRUE( nullptr == str1->append( cstr_append_fail ) );

    /* But adding a slightly shorter string should work: */

    ASSERT_TRUE( str1->getMaxLength() == std::strlen( cstr_after_append2 ) );
    ASSERT_TRUE( nullptr != str1->append( cstr_append2 ) );
    ASSERT_TRUE( str1->getMaxLength() == str1->getLength() );
    ASSERT_TRUE( str1->getMaxLength() == std::strlen( cstr_after_append2 ) );

    /* Compare results to the expected stored state */

    ASSERT_TRUE( std::strcmp( str1->getCString(), cstr_after_append2 ) == 0 );
    ASSERT_TRUE( str1->getLength() == std::strlen( cstr_after_append2 ) );
    ASSERT_TRUE( str1->getLength() == str1->getMaxLength() );
    ASSERT_TRUE( str1->getCapacity() == str1_capacity );
    ASSERT_TRUE( str1->getCapacity() >  str1->getMaxLength() );

    /* Appending any string to str2 does not work, however as str2 is already
     * at the maximum length */

    ASSERT_TRUE( str2->getLength() == str2->getMaxLength() );
    ASSERT_TRUE( nullptr == str2->append( cstr_append2 ) );
    ASSERT_TRUE( str2->getLength() == str2->getMaxLength() );

    /* Verify that clearing str1 results in an empty string stored */

    str1->clear();

    ASSERT_TRUE( str1->getLength() == buf_size_t{ 0 } );
    ASSERT_TRUE( std::strlen( str1->getCString() ) == buf_size_t{ 0 } );
    ASSERT_TRUE( str1->getCapacity() == str1_capacity );
    ASSERT_TRUE( str1->getMaxLength() == cstr_max_length );

    /* You can use the "regular" string functions with a NS(BufferStringObj);
     * but you have to manually sync its internal state whenevery the
     * length of the string has been altered */

    std::strncpy( str1->getCString(), str2->getCString(),
                  str1->getMaxLength() );

    str1->syncLength();

    ASSERT_TRUE( str1->getLength() == str2->getLength() );
    ASSERT_TRUE( 0 == std::strcmp( str1->getCString(), str2->getCString() ) );
}


/* end: tests/sixtracklib/common/test_buffer_string_object_cxx.cpp */
