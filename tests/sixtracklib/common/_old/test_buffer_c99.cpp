#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"

#include "sixtracklib/common/buffer.h"
#include "sixtracklib/testlib.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        struct MyObj
        {
            ::NS(object_type_id_t)   type_id  SIXTRL_ALIGN( 8u );
            int32_t                  a        SIXTRL_ALIGN( 8u );
            double                   b        SIXTRL_ALIGN( 8u );
            double                   c[ 4 ]   SIXTRL_ALIGN( 8u );
            uint8_t* SIXTRL_RESTRICT d        SIXTRL_ALIGN( 8u );
            double*  SIXTRL_RESTRICT e        SIXTRL_ALIGN( 8u );
        };
    }
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, InitOnExistingFlatMemory)
{
    std::vector< unsigned char > too_small( 36u, uint8_t{ 0 } );
    std::vector< unsigned char > data_buffer( ( 1u << 20u ), uint8_t{ 0 } );

    ::NS(Buffer) buffer;
    ::NS(Buffer_preset)( &buffer );

    int success = ::NS(Buffer_init_on_flat_memory)(
        &buffer, too_small.data(), too_small.size() );

    ASSERT_TRUE( success != 0 );

    ::NS(Buffer_preset)( &buffer );

    success = ::NS(Buffer_init_on_flat_memory)(
        &buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
}

TEST( C99_CommonBufferTests, NewOnExistingFlatMemory)
{
    std::vector< unsigned char > raw_buffer( 1u << 20u );

    ::NS(Buffer)* buffer = ::NS(Buffer_new_on_memory)(
        raw_buffer.data(), raw_buffer.size() * sizeof( unsigned char ) );

    ASSERT_TRUE(  buffer != nullptr );
    ASSERT_TRUE(  ::NS(Buffer_get_num_of_objects)( buffer ) == 0u );
    ASSERT_TRUE( !::NS(Buffer_owns_datastore)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_get_num_of_dataptrs)( buffer ) == 0u );
    ASSERT_TRUE(  ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) == 0u );
    ASSERT_TRUE(  ::NS(Buffer_get_size)( buffer ) > ::st_buffer_size_t{ 0 } );

    ::NS(Buffer_delete)( buffer );
    buffer = nullptr;
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, InitFlatMemoryDataStoreAddObjectsRemapAndCompare )
{
    namespace sixtrl = SIXTRL_CXX_NAMESPACE::tests;

    using my_obj_t   = sixtrl::MyObj;
    using obj_t      = ::NS(Object);
    using buf_size_t = ::NS(buffer_size_t);

    static double const EPS = std::numeric_limits< double >::epsilon();
    static constexpr buf_size_t MY_OBJ_SIZE = sizeof( my_obj_t );

    /* --------------------------------------------------------------------- */
    /* WRITING TO FLAT_MEMORY_BUFFER                                         */
    /* --------------------------------------------------------------------- */

    std::vector< unsigned char > data_buffer( 4096u, uint8_t{ 0 } );

    buf_size_t const N = 3u;
    constexpr buf_size_t NUM_DATAPTRS = 2u;

    buf_size_t const offsets[ NUM_DATAPTRS ] =
    {
        offsetof( my_obj_t, d ),
        offsetof( my_obj_t, e )
    };

    buf_size_t const type_sizes[ NUM_DATAPTRS ] =
    {
        sizeof( uint8_t ), sizeof( double )
    };

    buf_size_t attr_counts[ NUM_DATAPTRS ] =
    {
        st_buffer_size_t{ 0 },
        st_buffer_size_t{ 0 }
    };

    ::NS(Buffer) buffer;
    ::NS(Buffer_preset)( &buffer );

    int success = ::NS(Buffer_init_on_flat_memory_detailed)( &buffer,
            data_buffer.data(), data_buffer.size(), N, 48u, 8u, 1u );

    ASSERT_TRUE( success == 0 );

    constexpr buf_size_t num_d_values = 4;
    constexpr buf_size_t num_e_values = 2;

    uint8_t obj1_d_values[ num_d_values ] = { 31, 32, 33, 34 };
    double  obj1_e_values[ num_e_values ] = { 35.0, 36.0 };

    attr_counts[ 0 ] = num_d_values;
    attr_counts[ 1 ] = num_e_values;

    my_obj_t obj1;
    obj1.type_id = 3u;
    obj1.a       = 25;
    obj1.b       = 26.0;
    obj1.c[ 0 ]  = 27.0;
    obj1.c[ 1 ]  = 28.0;
    obj1.c[ 2 ]  = 29.0;
    obj1.c[ 3 ]  = 30.0;
    obj1.d       = &obj1_d_values[ 0 ];
    obj1.e       = &obj1_e_values[ 0 ];

    obj_t* ptr_object = ::NS(Buffer_add_object)( &buffer, &obj1,
        MY_OBJ_SIZE, obj1.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_object ) == obj1.type_id );
    ASSERT_TRUE( ::NS(Object_get_size)( ptr_object ) > MY_OBJ_SIZE );

    my_obj_t* ptr_stored_obj = reinterpret_cast< my_obj_t* >(
       ::NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj1 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj1.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj1.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj1.b ) <= EPS );

    for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj1.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj1.c[ ii ] ) <= EPS );
    }

    for( buf_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != &obj1.d[ ii ] );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  obj1.d[ ii ] );
    }

    for( buf_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != &obj1.e[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - obj1.e[ ii ] )
            <= EPS );
    }

    /* --------------------------------------------------------------------- */

    uint8_t obj2_d_values[ num_d_values ] =
    {
        uint8_t{ 106 }, uint8_t{ 107 }, uint8_t{ 108 }, uint8_t{ 109 }
    };

    double obj2_e_values[ num_e_values ] = { 110.0, 111.0 };

    my_obj_t obj2;
    obj2.type_id = 4u;
    obj2.a       = 100;
    obj2.b       = 101.0;
    obj2.c[ 0 ]  = 102.0;
    obj2.c[ 1 ]  = 103.0;
    obj2.c[ 2 ]  = 104.0;
    obj2.c[ 3 ]  = 105.0;
    obj2.d       = &obj2_d_values[ 0 ];
    obj2.e       = &obj2_e_values[ 0 ];

    ptr_object = ::NS(Buffer_add_object)( &buffer, &obj2, MY_OBJ_SIZE,
        obj2.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_object ) == obj2.type_id );
    ASSERT_TRUE( ::NS(Object_get_size)( ptr_object ) > MY_OBJ_SIZE );

    ptr_stored_obj = reinterpret_cast< my_obj_t* >(
        ::NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj1 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj2.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj2.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj2.b ) <= EPS );

    for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj2.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj2.c[ ii ] ) <= EPS );
    }

    for( buf_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != &obj2.d[ ii ] );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  obj2.d[ ii ] );
    }

    for( buf_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != &obj2.e[ ii ] );
        ASSERT_TRUE( std::abs( ptr_stored_obj->e[ ii ] - obj2.e[ ii ] ) <= EPS );
    }

    /* --------------------------------------------------------------------- */

    my_obj_t obj3;
    obj3.type_id = 5u;
    obj3.a       = 200;
    obj3.b       = 201.0;
    obj3.c[ 0 ]  = 202.0;
    obj3.c[ 1 ]  = 203.0;
    obj3.c[ 2 ]  = 204.0;
    obj3.c[ 3 ]  = 205.0;
    obj3.d       = nullptr;
    obj3.e       = nullptr;

    ptr_object = ::NS(Buffer_add_object)( &buffer, &obj3, MY_OBJ_SIZE,
        obj3.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_object ) == obj3.type_id );
    ASSERT_TRUE( ::NS(Object_get_size)( ptr_object ) > MY_OBJ_SIZE );

    ptr_stored_obj = reinterpret_cast< my_obj_t* >(
        ::NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj3 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj3.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj3.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj3.b ) <=
                 EPS );

    for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj3.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj3.c[ ii ] ) <= EPS );
    }

    for( buf_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != nullptr );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  uint8_t{ 0 } );
    }

    for( buf_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != nullptr );
        ASSERT_TRUE( std::abs( ptr_stored_obj->e[ ii ] - double{ 0.0 } ) <= EPS );
    }

    /* --------------------------------------------------------------------- */
    /* COPYING FLAT MEMORY BUFFER TO DIFFERENT MEMORY BUFFER ->
     * THIS SIMULATES THE EFFECTS OF MOVING THE BUFFER TO A DIFFERENT
     * MEMORY REALM, LIKE FOR EXAMPLE ON A GPU-TYPE DEVICE (OR READING
     * FROM A FILE, ETC. )*/
    /* --------------------------------------------------------------------- */

    std::vector< unsigned char > copy_buffer(
        NS(Buffer_get_size)( &buffer ), uint8_t{ 0 } );

    copy_buffer.assign( NS(Buffer_get_const_data_begin)( &buffer ),
                        NS(Buffer_get_const_data_end)( &buffer ) );

    ::NS(Buffer) cmp_buffer;
    ::NS(Buffer_preset)( &cmp_buffer );

    success = ::NS(Buffer_init_from_data)(
        &cmp_buffer, copy_buffer.data(), copy_buffer.size() );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ::NS(Buffer_get_size)( &buffer ) ==
                 ::NS(Buffer_get_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &cmp_buffer ) != 0u );
    ASSERT_TRUE( ::NS(Buffer_get_data_end_addr)(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &buffer ) !=
                 ::NS(Buffer_get_data_begin_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_slot_size)( &buffer ) ==
                 ::NS(Buffer_get_slot_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_header_size)( &buffer ) ==
                 ::NS(Buffer_get_header_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_begin_addr)( &cmp_buffer ) != 0u );
    ASSERT_TRUE( ::NS(Buffer_get_objects_end_addr)(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( ::NS(Buffer_get_objects_begin_addr)( &buffer ) !=
                 ::NS(Buffer_get_objects_begin_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_end_addr)( &buffer ) !=
                 ::NS(Buffer_get_objects_end_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( &buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ASSERT_TRUE( ::NS(Buffer_get_slots_size)( &buffer ) ==
                 ::NS(Buffer_get_slots_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( &buffer ) ==
                 ::NS(Buffer_get_num_of_slots)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_size)( &buffer ) ==
                 ::NS(Buffer_get_objects_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_dataptrs_size)( &buffer ) ==
                 ::NS(Buffer_get_dataptrs_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( &buffer ) ==
                 ::NS(Buffer_get_num_of_dataptrs)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_garbage_size)( &buffer ) ==
                 ::NS(Buffer_get_garbage_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( &buffer ) ==
                 ::NS(Buffer_get_num_of_garbage_ranges)( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_t const* obj_it  = ::NS(Buffer_get_const_objects_begin)( &buffer );
    obj_t const* obj_end = ::NS(Buffer_get_const_objects_end)( &buffer );

    obj_t const* cmp_obj_it  =
        ::NS(Buffer_get_const_objects_begin)( &cmp_buffer );

    obj_t const* cmp_obj_end =
        ::NS(Buffer_get_const_objects_end)( &cmp_buffer );

    ASSERT_TRUE( obj_it      != nullptr     );
    ASSERT_TRUE( obj_end     != nullptr     );
    ASSERT_TRUE( cmp_obj_it  != nullptr     );
    ASSERT_TRUE( cmp_obj_end != nullptr     );
    ASSERT_TRUE( obj_it      != cmp_obj_it  );
    ASSERT_TRUE( obj_end     != cmp_obj_end );

    ASSERT_TRUE( std::distance( obj_it, obj_end ) ==
                 std::distance( cmp_obj_it, cmp_obj_end ) );

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_obj_it )
    {
        ASSERT_TRUE( ::NS(Object_get_type_id)( obj_it ) ==
                     ::NS(Object_get_type_id)( cmp_obj_it ) );

        ASSERT_TRUE( ::NS(Object_get_size)( obj_it ) ==
                     ::NS(Object_get_size)( cmp_obj_it ) );
        ASSERT_TRUE( ::NS(Object_get_size)( obj_it ) > MY_OBJ_SIZE );

        ASSERT_TRUE( ::NS(Object_get_const_begin_ptr)( obj_it ) !=
                     ::NS(Object_get_const_begin_ptr)( cmp_obj_it ) );

        my_obj_t const* ptr_my_obj = reinterpret_cast< my_obj_t const* >(
            ::NS(Object_get_const_begin_ptr)( obj_it ) );

        my_obj_t const* ptr_cmp_obj = reinterpret_cast< my_obj_t const* >(
            ::NS(Object_get_const_begin_ptr)( cmp_obj_it ) );

        ASSERT_TRUE( ptr_my_obj  != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != ptr_my_obj );

        ASSERT_TRUE( ptr_cmp_obj->type_id == ptr_my_obj->type_id );
        ASSERT_TRUE( ptr_cmp_obj->a       == ptr_my_obj->a       );
        ASSERT_TRUE( std::fabs( ptr_cmp_obj->b - ptr_my_obj->b ) <= EPS );

        for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( &ptr_cmp_obj->c[ ii ] != &ptr_my_obj->c[ ii ] );
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->c[ ii ] -
                ptr_my_obj->c[ ii ] ) <= EPS );
        }

        ASSERT_TRUE( ptr_cmp_obj->d != ptr_my_obj->d );
        ASSERT_TRUE( ptr_cmp_obj->d != nullptr );
        ASSERT_TRUE( ptr_my_obj->d  != nullptr );

        for( buf_size_t ii = 0 ; ii < num_d_values ; ++ii )
        {
            ASSERT_TRUE( ptr_cmp_obj->d[ ii ] == ptr_my_obj->d[ ii ] );
        }

        ASSERT_TRUE( ptr_cmp_obj->e != ptr_my_obj->e );
        ASSERT_TRUE( ptr_cmp_obj->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e  != nullptr );

        for( buf_size_t ii = 0 ; ii < num_e_values ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->e[ ii ] -
                ptr_my_obj->e[ ii ] ) <= EPS );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Remap the same buffer again -> should work out just fine!             */
    /* --------------------------------------------------------------------- */

    ::NS(Buffer_preset)( &cmp_buffer );

    success = ::NS(Buffer_init_from_data)(
        &cmp_buffer, copy_buffer.data(), copy_buffer.size() );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ::NS(Buffer_get_size)( &buffer ) ==
                 ::NS(Buffer_get_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &cmp_buffer ) != 0u );
    ASSERT_TRUE( ::NS(Buffer_get_data_end_addr)(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &buffer ) !=
                 ::NS(Buffer_get_data_begin_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_slot_size)( &buffer ) ==
                 ::NS(Buffer_get_slot_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_header_size)( &buffer ) ==
                 ::NS(Buffer_get_header_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_begin_addr)( &cmp_buffer ) != 0u );
    ASSERT_TRUE( ::NS(Buffer_get_objects_end_addr)(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( ::NS(Buffer_get_objects_begin_addr)( &buffer ) !=
                 ::NS(Buffer_get_objects_begin_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_end_addr)( &buffer ) !=
                 ::NS(Buffer_get_objects_end_addr)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( &buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ASSERT_TRUE( ::NS(Buffer_get_slots_size)( &buffer ) ==
                 ::NS(Buffer_get_slots_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( &buffer ) ==
                 ::NS(Buffer_get_num_of_slots)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_objects_size)( &buffer ) ==
                 ::NS(Buffer_get_objects_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_dataptrs_size)( &buffer ) ==
                 ::NS(Buffer_get_dataptrs_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( &buffer ) ==
                 ::NS(Buffer_get_num_of_dataptrs)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_garbage_size)( &buffer ) ==
                 ::NS(Buffer_get_garbage_size)( &cmp_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( &buffer ) ==
                 ::NS(Buffer_get_num_of_garbage_ranges)( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_it  = ::NS(Buffer_get_const_objects_begin)( &buffer );
    obj_end = ::NS(Buffer_get_const_objects_end)(   &buffer );

    cmp_obj_it  = ::NS(Buffer_get_const_objects_begin)( &cmp_buffer );
    cmp_obj_end = ::NS(Buffer_get_const_objects_end)(   &cmp_buffer );

    ASSERT_TRUE( obj_it      != nullptr     );
    ASSERT_TRUE( obj_end     != nullptr     );
    ASSERT_TRUE( cmp_obj_it  != nullptr     );
    ASSERT_TRUE( cmp_obj_end != nullptr     );
    ASSERT_TRUE( obj_it      != cmp_obj_it  );
    ASSERT_TRUE( obj_end     != cmp_obj_end );

    ASSERT_TRUE( std::distance( obj_it, obj_end ) ==
                 std::distance( cmp_obj_it, cmp_obj_end ) );

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_obj_it )
    {
        ASSERT_TRUE( ::NS(Object_get_type_id)( obj_it ) ==
                     ::NS(Object_get_type_id)( cmp_obj_it ) );

        ASSERT_TRUE( ::NS(Object_get_size)( obj_it ) ==
                     ::NS(Object_get_size)( cmp_obj_it ) );

        ASSERT_TRUE( ::NS(Object_get_size)( obj_it ) > MY_OBJ_SIZE );
        ASSERT_TRUE( ::NS(Object_get_const_begin_ptr)( obj_it ) !=
                     ::NS(Object_get_const_begin_ptr)( cmp_obj_it ) );

        my_obj_t const* ptr_my_obj = reinterpret_cast< my_obj_t const* >(
            ::NS(Object_get_const_begin_ptr)( obj_it ) );

        my_obj_t const* ptr_cmp_obj = reinterpret_cast< my_obj_t const* >(
            ::NS(Object_get_const_begin_ptr)( cmp_obj_it ) );

        ASSERT_TRUE( ptr_my_obj  != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != ptr_my_obj );

        ASSERT_TRUE( ptr_cmp_obj->type_id == ptr_my_obj->type_id );
        ASSERT_TRUE( ptr_cmp_obj->a       == ptr_my_obj->a       );
        ASSERT_TRUE( std::fabs( ptr_cmp_obj->b - ptr_my_obj->b ) <= EPS );

        for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( &ptr_cmp_obj->c[ ii ] != &ptr_my_obj->c[ ii ] );
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->c[ ii ] -
                ptr_my_obj->c[ ii ] ) <= EPS );
        }

        ASSERT_TRUE( ptr_cmp_obj->d != ptr_my_obj->d );
        ASSERT_TRUE( ptr_cmp_obj->d != nullptr );
        ASSERT_TRUE( ptr_my_obj->d  != nullptr );

        for( buf_size_t ii = 0 ; ii < num_d_values ; ++ii )
        {
            ASSERT_TRUE( ptr_cmp_obj->d[ ii ] == ptr_my_obj->d[ ii ] );
        }

        ASSERT_TRUE( ptr_cmp_obj->e != ptr_my_obj->e );
        ASSERT_TRUE( ptr_cmp_obj->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e  != nullptr );

        for( buf_size_t ii = 0 ; ii < num_e_values ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->e[ ii ] -
                ptr_my_obj->e[ ii ] ) <= EPS );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Free Buffer -> not necessary, but should also do no harm! */

    ::NS(Buffer_free)( &buffer );
    ::NS(Buffer_free)( &cmp_buffer );
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, ReconstructFromCObjectFile )
{
    namespace sixtrl = SIXTRL_CXX_NAMESPACE::tests;

    using my_obj_t   = sixtrl::MyObj;
    using obj_t      = ::NS(Object);
    using buf_size_t = ::NS(buffer_size_t);

    static const double EPS = std::numeric_limits< double >::epsilon();
    static constexpr buf_size_t MY_OBJ_SIZE = sizeof( sixtrl::MyObj );

    /* --------------------------------------------------------------------- */
    /* PREPARE DATA OBJECTS                                                  */
    /* --------------------------------------------------------------------- */

    std::vector< unsigned char > data_buffer( 4096u, uint8_t{ 0 } );

    buf_size_t const N = 3u;
    constexpr buf_size_t NUM_DATAPTRS = 2u;

    buf_size_t const offsets[ NUM_DATAPTRS ] =
    {
        offsetof( my_obj_t, d ),
        offsetof( my_obj_t, e )
    };

    buf_size_t const type_sizes[ NUM_DATAPTRS ] =
    {
        sizeof( uint8_t ), sizeof( double )
    };

    buf_size_t attr_counts[ NUM_DATAPTRS ] =
    {
        st_buffer_size_t{ 0 },
        st_buffer_size_t{ 0 }
    };

    constexpr buf_size_t num_d_values = 4;
    constexpr buf_size_t num_e_values = 2;

    uint8_t obj1_d_values[ num_d_values ] = { 31, 32, 33, 34 };
    double  obj1_e_values[ num_e_values ] = { 35.0, 36.0 };

    attr_counts[ 0 ] = num_d_values;
    attr_counts[ 1 ] = num_e_values;

    my_obj_t cmp_my_obj[ 3 ];
    std::memset( &cmp_my_obj[ 0 ], ( int )0, MY_OBJ_SIZE * 3u );

    my_obj_t& obj1 = cmp_my_obj[ 0 ];
    obj1.type_id = 3u;
    obj1.a       = 25;
    obj1.b       = 26.0;
    obj1.c[ 0 ]  = 27.0;
    obj1.c[ 1 ]  = 28.0;
    obj1.c[ 2 ]  = 29.0;
    obj1.c[ 3 ]  = 30.0;
    obj1.d       = &obj1_d_values[ 0 ];
    obj1.e       = &obj1_e_values[ 0 ];

    /* --------------------------------------------------------------------- */

    uint8_t obj2_d_values[ num_d_values ] =
    {
        uint8_t{ 106 }, uint8_t{ 107 }, uint8_t{ 108 }, uint8_t{ 109 }
    };

    double obj2_e_values[ num_e_values ] = { 110.0, 111.0 };

    my_obj_t& obj2 = cmp_my_obj[ 1 ];
    obj2.type_id = 4u;
    obj2.a       = 100;
    obj2.b       = 101.0;
    obj2.c[ 0 ]  = 102.0;
    obj2.c[ 1 ]  = 103.0;
    obj2.c[ 2 ]  = 104.0;
    obj2.c[ 3 ]  = 105.0;
    obj2.d       = &obj2_d_values[ 0 ];
    obj2.e       = &obj2_e_values[ 0 ];

    /* --------------------------------------------------------------------- */

    my_obj_t& obj3 = cmp_my_obj[ 2 ];
    obj3.type_id = 5u;
    obj3.a       = 200;
    obj3.b       = 201.0;
    obj3.c[ 0 ]  = 202.0;
    obj3.c[ 1 ]  = 203.0;
    obj3.c[ 2 ]  = 204.0;
    obj3.c[ 3 ]  = 205.0;
    obj3.d       = nullptr;
    obj3.e       = nullptr;

    /* --------------------------------------------------------------------- */
    /* WRITE BUFFER TO BINARY FILE */
    /* --------------------------------------------------------------------- */

    char const PATH_TO_BINARY_FILE[] = "./test.np";
    FILE* fp = std::fopen( PATH_TO_BINARY_FILE, "rb" );

    if( fp == nullptr )
    {
        fp = std::fopen( PATH_TO_BINARY_FILE, "wb" );
        ASSERT_TRUE( fp != nullptr );

        ::NS(Buffer) buffer;
        ::NS(Buffer_preset)( &buffer );

        int success = ::NS(Buffer_init_on_flat_memory_detailed)( &buffer,
                data_buffer.data(), data_buffer.size(), N, 48u, 8u, 1u );

        ASSERT_TRUE( success == 0 );

        obj_t* ptr_object = ::NS(Buffer_add_object)( &buffer, &obj1,
            MY_OBJ_SIZE, obj1.type_id, NUM_DATAPTRS, &offsets[ 0 ],
                &type_sizes[ 0 ], &attr_counts[ 0 ] );

        ASSERT_TRUE( ptr_object != nullptr );

        ptr_object = ::NS(Buffer_add_object)( &buffer, &obj2, MY_OBJ_SIZE,
            obj2.type_id, NUM_DATAPTRS, &offsets[ 0 ], &type_sizes[ 0 ],
                &attr_counts[ 0 ] );

        ASSERT_TRUE( ptr_object != nullptr );

        buf_size_t const cnt = std::fwrite( ( unsigned char const* )(
            uintptr_t )::NS(Buffer_get_data_begin_addr)( &buffer ),
            ::NS(Buffer_get_size)( &buffer ), st_buffer_size_t{ 1 }, fp );

        ASSERT_TRUE( cnt == st_buffer_size_t{ 1 } );

        std::fclose( fp );
        fp = nullptr;
    }
    else
    {
        std::fclose( fp );
        fp = nullptr;
    }

    ASSERT_TRUE( fp == nullptr );

    fp = std::fopen( PATH_TO_BINARY_FILE, "rb" );
    ASSERT_TRUE( fp != nullptr );

    std::fseek( fp, 0, SEEK_END );
    long int const fp_end_pos = std::ftell( fp );

    std::fseek( fp, 0, SEEK_SET );
    long int const fp_begin_pos = std::ftell( fp );

    std::size_t const buffer_size = ( fp_end_pos >= fp_begin_pos )
        ? static_cast< std::size_t >( fp_end_pos - fp_begin_pos )
        : std::size_t{ 0 };

    std::vector< unsigned char > base_buffer( buffer_size, uint8_t{ 0 } );
    ASSERT_TRUE( buffer_size > std::size_t{ 0 } );

    unsigned char* data_buffer_begin = base_buffer.data();
    ASSERT_TRUE( data_buffer_begin != nullptr );

    std::size_t cnt = std::fread( data_buffer_begin, buffer_size, 1u, fp );
    ASSERT_TRUE( cnt == 1u );

    if( fp != nullptr )
    {
        std::fclose( fp );
        fp = nullptr;
    }

    ASSERT_TRUE( fp == nullptr );
    std::remove( PATH_TO_BINARY_FILE );

    ::NS(Buffer) buffer;
    ::NS(Buffer_preset)( &buffer );

    int success = ::NS(Buffer_init_from_data)(
        &buffer, data_buffer_begin, buffer_size );

    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( ::NS(Buffer_get_size)( &buffer ) == buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( &buffer ) == 2u );

    st_Object const* obj_it  = ::NS(Buffer_get_const_objects_begin)(&buffer);
    st_Object const* obj_end = ::NS(Buffer_get_const_objects_end)( &buffer );

    my_obj_t const* cmp_obj_it = &cmp_my_obj[ 0 ];

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_obj_it )
    {
        using ptr_my_obj_t = my_obj_t const*;

        ASSERT_TRUE( ::NS(Object_get_type_id)( obj_it ) ==
                     cmp_obj_it->type_id );

        ASSERT_TRUE( ::NS(Object_get_size)( obj_it ) > MY_OBJ_SIZE );
        ASSERT_TRUE( ::NS(Object_get_const_begin_ptr)( obj_it ) != nullptr );

        ptr_my_obj_t ptr_my_obj = reinterpret_cast< ptr_my_obj_t >(
                ::NS(Object_get_const_begin_ptr)( obj_it ) );

        ASSERT_TRUE( ptr_my_obj != nullptr );
        ASSERT_TRUE( ptr_my_obj->type_id ==
                     ::NS(Object_get_type_id)( obj_it ) );

        ASSERT_TRUE( ptr_my_obj->a == cmp_obj_it->a );
        ASSERT_TRUE( std::abs( ( ptr_my_obj->b - cmp_obj_it->b ) < EPS ) );

        for( std::size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ( ptr_my_obj->c[ ii ] -
                cmp_obj_it->c[ ii ] ) < EPS ) );
        }

        ASSERT_TRUE( ptr_my_obj->d != nullptr );
        ASSERT_TRUE( cmp_my_obj->d != nullptr );

        for( std::size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( std::abs( ( ptr_my_obj->d[ ii ] -
                cmp_obj_it->d[ ii ] ) < EPS ) );
        }

        ASSERT_TRUE( ptr_my_obj->e != nullptr );
        ASSERT_TRUE( cmp_obj_it->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e[ 0 ] == cmp_obj_it->e[ 0 ] );
        ASSERT_TRUE( ptr_my_obj->e[ 1 ] == cmp_obj_it->e[ 1 ] );
    }

    ::NS(Buffer_free)( &buffer );
}

TEST( C99_CommonBufferTests, NewBufferAndGrowingWithinCapacity )
{
    namespace sixtrl = SIXTRL_CXX_NAMESPACE::tests;

    using my_obj_t   = sixtrl::MyObj;
    using obj_t      = ::NS(Object);
    using buf_size_t = ::NS(buffer_size_t);
    using type_id_t  = ::NS(object_type_id_t);

    static constexpr buf_size_t ZERO = buf_size_t{ 0u };
    static constexpr buf_size_t MY_OBJ_SIZE = sizeof( my_obj_t );

    NS(Buffer)* buffer = ::NS(Buffer_new)( buf_size_t{ 1u << 20u } );

    ASSERT_TRUE( buffer != nullptr );

    ASSERT_TRUE(  ::NS(Buffer_has_datastore)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_uses_datastore)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_owns_datastore)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_allow_modify_datastore_contents)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_allow_clear)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_allow_append_objects)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_allow_remapping)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_allow_resize)( buffer ) );
    ASSERT_TRUE(  ::NS(Buffer_uses_mempool_datastore)( buffer ) );
    ASSERT_TRUE( !::NS(Buffer_uses_special_opencl_datastore)( buffer ) );
    ASSERT_TRUE( !::NS(Buffer_uses_special_cuda_datastore)( buffer ) );

    buf_size_t const slot_size = ::NS(Buffer_get_slot_size)( buffer );
    ASSERT_TRUE( slot_size > ZERO );

    buf_size_t const sect_hdr_len =
        ::NS(Buffer_get_section_header_size)( buffer );

    ASSERT_TRUE( ::NS(Buffer_get_size)( buffer ) > ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_slots)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_slots_size)( buffer ) == sect_hdr_len );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_objects)(buffer) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_objects_size)( buffer ) == sect_hdr_len );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_dataptrs)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_dataptrs_size)( buffer ) == sect_hdr_len );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_garbage_ranges)( buffer ) == ZERO );
    ASSERT_TRUE( ::NS(Buffer_get_garbage_size)( buffer ) == sect_hdr_len );

    /* --------------------------------------------------------------------- *
     * ADD A MYOBJ INSTANCE TO THE BUFFER -> THE BUFFER HAS TO GROW          *
     * --------------------------------------------------------------------- */

    constexpr buf_size_t num_d_values = buf_size_t{ 16000 };
    constexpr buf_size_t num_e_values = buf_size_t{ 8000  };
    constexpr buf_size_t num_dataptrs = buf_size_t{ 2u };

    buf_size_t const uint8_size  = sizeof( uint8_t );
    buf_size_t const double_size = sizeof( double );

    buf_size_t const attr_offsets[ num_dataptrs ] =
    {
        offsetof( my_obj_t, d ),
        offsetof( my_obj_t, e )
    };

    buf_size_t attr_sizes[  num_dataptrs ] = { uint8_size, double_size };
    buf_size_t attr_counts[ num_dataptrs ] = { num_d_values, num_e_values };

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
     * Add first obj -> obj1
     * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    my_obj_t obj1;
    std::memset( &obj1, ( int )0, MY_OBJ_SIZE );

    obj1.type_id   = 3u;
    obj1.a         = 1;
    obj1.b         = 2.0;
    obj1.c[ 0 ]    = 3.0;
    obj1.c[ 1 ]    = 4.0;
    obj1.c[ 2 ]    = 5.0;
    obj1.c[ 3 ]    = 6.0;
    obj1.d         = nullptr;
    obj1.e         = nullptr;

    obj_t* ptr_object = ::NS(Buffer_add_object)( buffer, &obj1, MY_OBJ_SIZE,
        static_cast< type_id_t >( obj1.type_id ), num_dataptrs, attr_offsets,
            attr_sizes, attr_counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) == buf_size_t{ 1u } );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) > ZERO );

    ASSERT_TRUE( ::NS(Object_get_size)( ptr_object ) >= MY_OBJ_SIZE );
    ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_object ) ==
                 static_cast< type_id_t >( obj1.type_id ) );

    ASSERT_TRUE( ::NS(Object_get_begin_ptr)( ptr_object ) != nullptr );

    my_obj_t* ptr_my_obj = reinterpret_cast< my_obj_t* >(
        ::NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_my_obj   != nullptr );
    ASSERT_TRUE( obj1.type_id == ptr_my_obj->type_id );
    ASSERT_TRUE( obj1.a       == ptr_my_obj->a       );

    ASSERT_TRUE( std::fabs( obj1.b - ptr_my_obj->b ) <
        std::numeric_limits< double >::epsilon() );

    for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( std::fabs( obj1.c[ ii ] - ptr_my_obj->c[ ii ] ) <
            std::numeric_limits< double >::epsilon() );
    }

    ASSERT_TRUE( ptr_my_obj->d != nullptr );
    ASSERT_TRUE( ptr_my_obj->e != nullptr );

    std::ptrdiff_t dist_d_to_e = std::distance(
        reinterpret_cast< unsigned char const* >( ptr_my_obj->d ),
        reinterpret_cast< unsigned char const* >( ptr_my_obj->e ) );

    ASSERT_TRUE( dist_d_to_e > 0 );
    ASSERT_TRUE( static_cast< size_t >( dist_d_to_e ) >=
        ::NS(ManagedBuffer_get_slot_based_length)(
            num_d_values * sizeof( uint8_t ), slot_size ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
     * Add second obj -> obj2
     * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    size_t const num_objects_after_obj1 =
        ::NS(Buffer_get_num_of_objects)( buffer );

    size_t const num_slots_after_obj1    =
        ::NS(Buffer_get_num_of_slots)( buffer );

    size_t const num_dataptrs_after_obj1 =
        ::NS(Buffer_get_num_of_dataptrs)( buffer );

    size_t const num_garbage_ranges_after_obj1 =
        ::NS(Buffer_get_num_of_garbage_ranges)( buffer );

    my_obj_t obj2;
    std::memset( &obj2, ( int )0, MY_OBJ_SIZE );

    obj2.type_id   = 9u;
    obj2.a         = 10;
    obj2.b         = 12.0;
    obj2.c[ 0 ]    = 13.0;
    obj2.c[ 1 ]    = 14.0;
    obj2.c[ 2 ]    = 15.0;
    obj2.c[ 3 ]    = 16.0;
    obj2.d         = nullptr;
    obj2.e         = nullptr;

    ptr_object = ::NS(Buffer_add_object)( buffer, &obj2, MY_OBJ_SIZE,
        static_cast< type_id_t >( obj2.type_id ), num_dataptrs, attr_offsets,
            attr_sizes, attr_counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
        ( num_objects_after_obj1 + 1u ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) >
        ( num_slots_after_obj1 ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
        ( num_dataptrs_after_obj1 + num_dataptrs ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
                 num_garbage_ranges_after_obj1 );

    ASSERT_TRUE( ::NS(Object_get_size)( ptr_object ) >= MY_OBJ_SIZE );
    ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_object ) ==
                 static_cast< type_id_t >( obj2.type_id ) );

    ASSERT_TRUE( ::NS(Object_get_begin_ptr)( ptr_object ) != nullptr );

    unsigned char const* ptr_obj1_e =
        reinterpret_cast< unsigned char const* >( ptr_my_obj->e );

    ptr_my_obj = reinterpret_cast< my_obj_t* >(
        ::NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_my_obj != nullptr );

    std::ptrdiff_t dist_obj1_e_to_obj2 = std::distance( ptr_obj1_e,
        reinterpret_cast< unsigned char const* >( ptr_my_obj ) );

    ASSERT_TRUE( dist_obj1_e_to_obj2 > 0 );
    ASSERT_TRUE( static_cast< size_t >( dist_obj1_e_to_obj2 ) >=
        ::NS(ManagedBuffer_get_slot_based_length)(
            num_e_values * sizeof( double ), slot_size ) );

    ASSERT_TRUE( obj2.type_id == ptr_my_obj->type_id );
    ASSERT_TRUE( obj2.a       == ptr_my_obj->a       );

    ASSERT_TRUE( std::fabs( obj2.b - ptr_my_obj->b ) <
        std::numeric_limits< double >::epsilon() );

    for( buf_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( std::fabs( obj2.c[ ii ] - ptr_my_obj->c[ ii ] ) <
            std::numeric_limits< double >::epsilon() );
    }

    ASSERT_TRUE( ptr_my_obj->d != nullptr );
    ASSERT_TRUE( ptr_my_obj->e != nullptr );

    dist_d_to_e = std::distance(
        reinterpret_cast< unsigned char const* >( ptr_my_obj->d ),
        reinterpret_cast< unsigned char const* >( ptr_my_obj->e ) );

    ASSERT_TRUE( dist_d_to_e > 0 );
    ASSERT_TRUE( static_cast< size_t >( dist_d_to_e ) >=
        ::NS(ManagedBuffer_get_slot_based_length)(
            num_d_values * sizeof( uint8_t ), slot_size ) );

    /* --------------------------------------------------------------------- *
     * CLEANUP & RESSOURCE MANAGEMENT                                        *
     * --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( buffer );
    buffer = nullptr;
}

TEST( C99_CommonBufferTests, AddGenericObjectsTestAutoGrowingOfBuffer )
{
    using buf_size_t    = ::NS(buffer_size_t);
    using type_id_t     = ::NS(object_type_id_t);
    using generic_obj_t = ::NS(GenericObj);

    ::NS(Buffer)* buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_allow_resize)( buffer ) );
    ASSERT_TRUE( ::NS(Buffer_allow_remapping)( buffer ) );
    ASSERT_TRUE( ::NS(Buffer_allow_append_objects)( buffer ) );

    buf_size_t prev_capacity     = ::NS(Buffer_get_capacity)( buffer );
    buf_size_t prev_size         = ::NS(Buffer_get_size)( buffer );
    buf_size_t prev_num_objects  = ::NS(Buffer_get_num_of_objects)( buffer );
    buf_size_t prev_num_slots    = ::NS(Buffer_get_num_of_slots)( buffer );
    buf_size_t prev_num_dataptrs = ::NS(Buffer_get_num_of_dataptrs)( buffer );

    ASSERT_TRUE( buffer            != nullptr );
    ASSERT_TRUE( prev_size         >  buf_size_t{ 0 } );
    ASSERT_TRUE( prev_capacity     >= prev_size );
    ASSERT_TRUE( prev_num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_dataptrs == buf_size_t{ 0 } );

    constexpr buf_size_t NUM_OBJ_TO_ADD = 100;
    constexpr buf_size_t num_d_values = 10;
    constexpr buf_size_t num_e_values = 10;

    buf_size_t ii = 0;

    for( ; ii < NUM_OBJ_TO_ADD ; ++ii )
    {
        generic_obj_t* obj = ::NS(GenericObj_new)( buffer,
            static_cast< type_id_t >( ii ), num_d_values, num_e_values );

        ASSERT_TRUE( obj != nullptr );

        buf_size_t const capacity = ::NS(Buffer_get_capacity)( buffer );
        buf_size_t const size     = ::NS(Buffer_get_size)( buffer );

        ASSERT_TRUE( capacity >= prev_capacity );
        ASSERT_TRUE( size     >  prev_size     );
        ASSERT_TRUE( capacity >= size  );

        prev_capacity = capacity;
        prev_size     = size;

        buf_size_t const num_objects  =
            ::NS(Buffer_get_num_of_objects)( buffer );

        buf_size_t const num_slots    =
            ::NS(Buffer_get_num_of_slots)( buffer );

        buf_size_t const num_dataptrs =
            ::NS(Buffer_get_num_of_dataptrs)( buffer );

        ASSERT_TRUE( num_objects  == prev_num_objects + buf_size_t{ 1 } );
        ASSERT_TRUE( num_slots    >= prev_num_slots );
        ASSERT_TRUE( num_dataptrs >= prev_num_dataptrs );

        prev_num_objects  = num_objects;
        prev_num_slots    = num_slots;
        prev_num_dataptrs = num_dataptrs;
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
        NUM_OBJ_TO_ADD );

    ::NS(Buffer_delete)( buffer );
}

TEST( C99_CommonBufferTests, WriteBufferNormalizedAddrRestoreVerify )
{
    using prng_seed_t   = unsigned long long;
    using buf_size_t    = ::NS(buffer_size_t);
    using address_t     = ::NS(buffer_addr_t);
    using gen_obj_t     = ::NS(GenericObj);
    using ptr_gen_obj_t = gen_obj_t const*;
    using ptr_obj_t     = ::NS(Object) const*;

    prng_seed_t const seed = prng_seed_t{ 20181105 };
    ::NS(Random_init_genrand64)( seed );

    ::NS(Buffer)* cmp_buffer  = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* temp_buffer = ::NS(Buffer_new)( 0u );

    ASSERT_TRUE( cmp_buffer  != nullptr );
    ASSERT_TRUE( temp_buffer != nullptr );

    buf_size_t const NUM_OBJ      = buf_size_t{ 100 };
    buf_size_t const num_d_values = buf_size_t{ 10  };
    buf_size_t const num_e_values = buf_size_t{ 10  };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJ ; ++ii )
    {
        ::NS(object_type_id_t) const type_id =
            static_cast< ::NS(object_type_id_t) >( ii );

        gen_obj_t* obj = ::NS(GenericObj_new)(
            cmp_buffer, type_id, num_d_values, num_e_values );

        ASSERT_TRUE( obj != nullptr );
        ::NS(GenericObj_init_random)( obj );

        gen_obj_t* copy_obj = ::NS(GenericObj_add_copy)( temp_buffer, obj );

        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != obj );
    }

    double const ABS_TRESHOLD = std::numeric_limits< double >::epsilon();

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( cmp_buffer  ) == NUM_OBJ );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( temp_buffer ) == NUM_OBJ );

    ASSERT_TRUE( ::NS(Buffer_get_size)( cmp_buffer ) ==
                 ::NS(Buffer_get_size)( temp_buffer ) );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJ ; ++ii )
    {
        ptr_obj_t orig_ptr = ::NS(Buffer_get_const_object)( cmp_buffer, ii );
        ptr_obj_t copy_ptr = ::NS(Buffer_get_const_object)( temp_buffer, ii );

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != orig_ptr );

        ptr_gen_obj_t orig_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( orig_ptr ) );

        ptr_gen_obj_t copy_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( copy_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::NS(GenericObj_compare_values)( orig_obj, copy_obj ) ) ||
            ( 0 == ::NS(GenericObj_compare_values_with_treshold)(
                orig_obj, copy_obj, ABS_TRESHOLD ) ) );
    }

    /* Write file and remap the contents to a normalized base address: */

    address_t const base_addr =
        ::NS(Buffer_get_data_begin_addr)( temp_buffer );

    address_t const target_addr = ( base_addr != address_t { 0x1000 } )
        ? address_t { 0x1000 } : address_t { 0x2000 };

    std::string const path_to_temp_file( "./temp_norm_addr.bin" );

    ASSERT_TRUE( ::NS(Buffer_write_to_file_normalized_addr)( temp_buffer,
        path_to_temp_file.c_str(), target_addr ) );

    /* repeat the previous checks to verify that the normaliezd write to a
     * file has not changed temp_buffer */

    ASSERT_TRUE( base_addr == ::NS(Buffer_get_data_begin_addr)( temp_buffer ) );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( temp_buffer ) == NUM_OBJ );

    ASSERT_TRUE( ::NS(Buffer_get_size)( cmp_buffer ) ==
                 ::NS(Buffer_get_size)( temp_buffer ) );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJ ; ++ii )
    {
        ptr_obj_t orig_ptr = ::NS(Buffer_get_const_object)( cmp_buffer, ii );
        ptr_obj_t copy_ptr = ::NS(Buffer_get_const_object)( temp_buffer, ii );

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != orig_ptr );

        ptr_gen_obj_t orig_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( orig_ptr ) );

        ptr_gen_obj_t copy_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( copy_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::NS(GenericObj_compare_values)( orig_obj, copy_obj ) ) ||
            ( 0 == ::NS(GenericObj_compare_values_with_treshold)(
                orig_obj, copy_obj, ABS_TRESHOLD ) ) );
    }

    /* First open the binary file and verify that the base address has been
     * changed to the target address successfully */

    FILE* fp = std::fopen( path_to_temp_file.c_str(), "rb" );
    ASSERT_TRUE( fp != nullptr );

    address_t stored_base_address = address_t{ 0 };

    std::size_t const cnt = std::fread(
        &stored_base_address, sizeof( address_t ), 1u, fp );

    ASSERT_TRUE( cnt == std::size_t{ 1 } );
    ASSERT_TRUE( stored_base_address == target_addr );

    std::fclose( fp );
    fp = nullptr;

    /* Then restore from the normalized dump and verify that the restored
     * buffer content is equal to the original cmp_buffer objects */

    ::NS(Buffer)* restored_buffer = ::NS(Buffer_new_from_file)(
        path_to_temp_file.c_str() );

    std::remove( path_to_temp_file.c_str() );

    ASSERT_TRUE( restored_buffer != nullptr );
    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( restored_buffer ) );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( restored_buffer ) == NUM_OBJ );

    ASSERT_TRUE( ::NS(Buffer_get_size)( cmp_buffer ) ==
                 ::NS(Buffer_get_size)( restored_buffer ) );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJ ; ++ii )
    {
        ptr_obj_t orig_ptr = ::NS(Buffer_get_const_object)( cmp_buffer, ii );
        ptr_obj_t rest_ptr = ::NS(Buffer_get_const_object)(
            restored_buffer, ii );

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( rest_ptr != nullptr );
        ASSERT_TRUE( rest_ptr != orig_ptr );

        ptr_gen_obj_t orig_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( orig_ptr ) );

        ptr_gen_obj_t rest_obj = reinterpret_cast< ptr_gen_obj_t >(
            ::NS(Object_get_const_begin_ptr)( rest_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( rest_obj != nullptr );
        ASSERT_TRUE( rest_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::NS(GenericObj_compare_values)( orig_obj, rest_obj ) ) ||
            ( 0 == ::NS(GenericObj_compare_values_with_treshold)(
                orig_obj, rest_obj, ABS_TRESHOLD ) ) );
    }

    ::NS(Buffer_delete)( cmp_buffer );
    ::NS(Buffer_delete)( temp_buffer );
    ::NS(Buffer_delete)( restored_buffer );
}

TEST( C99_CommonBufferTests, CreateNewOnDataAddObjects )
{
    using gen_obj_t  = ::NS(GenericObj);
    using buf_size_t = ::NS(buffer_size_t);
    using i32_t      = SIXTRL_INT32_T;
    using u8_t       = SIXTRL_UINT8_T;
    using real_t     = SIXTRL_REAL_T;
    using type_id_t  = ::NS(object_type_id_t);

    /* --------------------------------------------------------------------- */
    /* Prepare dummy ::NS(GenericObj) instances to be added to the buffers */

    static constexpr buf_size_t TOTAL_NUM_OBJ = buf_size_t{ 10 };

    /* Use somewhat uneven numbers to implcitily also force addiitional
     * checks concerning alignment */
    buf_size_t const num_d_values = buf_size_t{ 97  };
    buf_size_t const num_e_values = buf_size_t{ 111 };

    /* --------------------------------------------------------------------- */
    /* Create an "external buffer": this one will provide the data array
     * for the "mapped" buffer later on and will have enough space
     * reserved for TOTAL_NUM_OBJ ::NS(GenericObj) instances: */

    ::NS(Buffer)* ext_buffer = ::NS(Buffer_new)( buf_size_t{ 0u } );

    buf_size_t num_objects  = buf_size_t{ 0 };
    buf_size_t num_slots    = buf_size_t{ 0 };
    buf_size_t num_dataptrs = buf_size_t{ 0 };
    buf_size_t num_garbage  = buf_size_t{ 0 };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < TOTAL_NUM_OBJ ; ++ii )
    {
        ++num_objects;

        num_slots += ::NS(GenericObj_predict_required_num_slots)(
            ext_buffer, num_d_values, num_e_values );

        num_dataptrs += ::NS(GenericObj_predict_required_num_dataptrs)(
            ext_buffer, num_d_values, num_e_values );
    }

    ASSERT_TRUE( num_objects  == TOTAL_NUM_OBJ );
    ASSERT_TRUE( num_slots    >  buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs == TOTAL_NUM_OBJ * buf_size_t{ 2u } );
    ASSERT_TRUE( num_garbage  == buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_reserve)( ext_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage ) == 0 );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_objects)( ext_buffer ) >=
        num_objects );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_slots)( ext_buffer ) >=
        num_slots );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_dataptrs)( ext_buffer ) >=
        num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_garbage_ranges)( ext_buffer ) >=
        num_garbage );

    /* --------------------------------------------------------------------- */
    /* Add half of the objects already to the ext_buffer; this allows us to
     * verify that the mapping buffer does not destroy the contents of the
     * ext_buffer and respects the capacity and sizes determiend here */

    static real_t const EPS = std::numeric_limits< real_t >::epsilon();
    buf_size_t const ADD_NUM_OBJS = TOTAL_NUM_OBJ / buf_size_t{ 2 };

    std::vector< u8_t   > d_values( num_d_values, u8_t{ 0 } );
    std::vector< real_t > e_values( num_e_values, real_t{ 0 } );

    std::vector< std::vector< u8_t   > > saved_d_values( TOTAL_NUM_OBJ );
    std::vector< std::vector< real_t > > saved_e_values( TOTAL_NUM_OBJ );

    u8_t   next_d_start_value = u8_t { 0 };
    real_t next_e_start_value = real_t{ 0 };

    real_t const c_values[] = {
            real_t{ 0 }, real_t{ 1 }, real_t{ 2 }, real_t{ 3 } };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < ADD_NUM_OBJS ; ++ii )
    {
        std::fill( d_values.begin(), d_values.end(), next_d_start_value );
        ++next_d_start_value;
        saved_d_values[ ii ] = d_values;

        std::iota( e_values.begin(), e_values.end(), next_e_start_value );
        next_e_start_value = e_values.back() + real_t{ 1 };
        saved_e_values[ ii ] = e_values;

        gen_obj_t* obj = ::NS(GenericObj_add)( ext_buffer,
            type_id_t{ 1 }, num_d_values, num_e_values,
            static_cast< i32_t >( ii ), static_cast< real_t >( ii ),
            &c_values[ 0 ], d_values.data(), e_values.data() );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ext_buffer ) ==
                     ( ii + buf_size_t{ 1 } ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( ext_buffer ) > 0u );
        ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( ext_buffer ) <
                     ::NS(Buffer_get_max_num_of_slots)( ext_buffer ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( ext_buffer ) ==
            ( ::NS(GenericObj_predict_required_num_dataptrs)(
                ext_buffer, num_d_values, num_e_values ) *
              ::NS(Buffer_get_num_of_objects)( ext_buffer ) ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( ext_buffer ) ==
                     buf_size_t{ 0 } );

        ASSERT_TRUE( obj != nullptr );
        ASSERT_TRUE( obj->a == static_cast< i32_t >( ii ) );
        ASSERT_TRUE( std::abs( obj->b - static_cast< real_t >( ii ) ) < EPS );
        ASSERT_TRUE( std::equal( &c_values[ 0 ], &c_values[ 4 ],   obj->c ) );

        ASSERT_TRUE( obj->num_d == num_d_values );
        ASSERT_TRUE( obj->d != nullptr );
        ASSERT_TRUE( std::equal( d_values.begin(), d_values.end(), obj->d ) );

        ASSERT_TRUE( obj->num_e == num_e_values );
        ASSERT_TRUE( obj->e != nullptr );
        ASSERT_TRUE( std::equal( e_values.begin(), e_values.end(), obj->e ) );
    }

    /* --------------------------------------------------------------------- */
    /* Generate a "mapped" buffer on the data array provided by ext_buffer   */

    unsigned char* ptr_ext_buffer_data = reinterpret_cast< unsigned char* >(
        static_cast< uintptr_t >( ::NS(Buffer_get_data_begin_addr)(
            ext_buffer ) ) );

    buf_size_t const ext_buffer_capacity =
        ::NS(Buffer_get_capacity)( ext_buffer );

    ::NS(Buffer)* buffer = ::NS(Buffer_new_on_data)(
        ptr_ext_buffer_data, ext_buffer_capacity );

    ASSERT_TRUE( buffer != nullptr );
    ASSERT_TRUE( buffer != ext_buffer );

    /* --------------------------------------------------------------------- */
    /* The expectation is that both buffers show precisely the same
     * content but ext_buffer "owns" the data buffer while
     * the mapped buffer merely acts as a view onto the data */

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( buffer ) ==
                 ::NS(Buffer_get_data_begin_addr)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( buffer ) ==
                 ::NS(Buffer_get_capacity)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( buffer ) ==
                 ::NS(Buffer_get_size)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_objects)( buffer ) ==
                 ::NS(Buffer_get_max_num_of_objects)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_slots)( buffer ) ==
                 ::NS(Buffer_get_max_num_of_slots)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_dataptrs)( buffer ) ==
                 ::NS(Buffer_get_max_num_of_dataptrs)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_garbage_ranges)( buffer ) ==
                 ::NS(Buffer_get_max_num_of_garbage_ranges)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) ==
                 ::NS(Buffer_get_num_of_slots)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
                 ::NS(Buffer_get_num_of_dataptrs)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
                 ::NS(Buffer_get_num_of_garbage_ranges)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_has_datastore)(  ext_buffer ) );
    ASSERT_TRUE( ::NS(Buffer_uses_datastore)( ext_buffer ) );
    ASSERT_TRUE( ::NS(Buffer_owns_datastore)( ext_buffer ) );

    ASSERT_TRUE( !::NS(Buffer_has_datastore)(  buffer ) );
    ASSERT_TRUE( !::NS(Buffer_uses_datastore)( buffer ) );
    ASSERT_TRUE( !::NS(Buffer_owns_datastore)( buffer ) );

    /* --------------------------------------------------------------------- */
    /* Add remaining objects via mapped buffer -> they should also
     * appear on the ext_buffer after calling NS(Buffer_refresh) */

    for( buf_size_t ii = ADD_NUM_OBJS ; ii < TOTAL_NUM_OBJ ; ++ii )
    {
        std::fill( d_values.begin(), d_values.end(), next_d_start_value );
        ++next_d_start_value;
        saved_d_values[ ii ] = d_values;

        std::iota( e_values.begin(), e_values.end(), next_e_start_value );
        next_e_start_value = e_values.back() + real_t{ 1 };
        saved_e_values[ ii ] = e_values;

        gen_obj_t* obj = ::NS(GenericObj_add)( buffer,
            type_id_t{ 1 }, num_d_values, num_e_values,
            static_cast< i32_t >( ii ), static_cast< real_t >( ii ),
            &c_values[ 0 ], d_values.data(), e_values.data() );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                     ( ii + buf_size_t{ 1 } ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) <=
                     ::NS(Buffer_get_max_num_of_objects)( buffer ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) > 0u );
        ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) <=
                     ::NS(Buffer_get_max_num_of_slots)( buffer ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
            ( ::NS(GenericObj_predict_required_num_dataptrs)(
                buffer, num_d_values, num_e_values ) *
              ::NS(Buffer_get_num_of_objects)( buffer ) ) );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
                     buf_size_t{ 0 } );

        ASSERT_TRUE( obj != nullptr );
        ASSERT_TRUE( obj->a == static_cast< i32_t >( ii ) );
        ASSERT_TRUE( std::abs( obj->b - static_cast< real_t >( ii ) ) < EPS );
        ASSERT_TRUE( std::equal( &c_values[ 0 ], &c_values[ 4 ],   obj->c ) );

        ASSERT_TRUE( obj->num_d == num_d_values );
        ASSERT_TRUE( obj->d != nullptr );
        ASSERT_TRUE( std::equal( d_values.begin(), d_values.end(), obj->d ) );

        ASSERT_TRUE( obj->num_e == num_e_values );
        ASSERT_TRUE( obj->e != nullptr );
        ASSERT_TRUE( std::equal( e_values.begin(), e_values.end(), obj->e ) );
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) == TOTAL_NUM_OBJ );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                 ::NS(Buffer_get_max_num_of_objects)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) > buf_size_t{ 0 } );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
        buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
        ::NS(GenericObj_predict_required_num_dataptrs)(
            buffer, num_d_values, num_e_values ) *
        ::NS(Buffer_get_num_of_objects)( buffer ) );

    /* --------------------------------------------------------------------- */
    /* We expect that the ext_buffer has gone out of sync since we added the *
     * the new instances via buffer rather than directly via ext_buffer.
     * Verify this: */

    /* NOTE: We have to !!temporarily!! set the ext_buffer in the
     * developer/debug mode since this discrepancy is checked against
     * at several stages! */

    ASSERT_TRUE( !::NS(Buffer_is_in_developer_debug_mode)( ext_buffer ) );
    ::NS(Buffer_enable_developer_debug_mode)( ext_buffer );
    ASSERT_TRUE(  ::NS(Buffer_is_in_developer_debug_mode)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) >
                 ext_buffer->num_objects );

    /* NOTE: Switch the developer/debug mode back off for ext_buffer */
    ::NS(Buffer_disable_developer_debug_mode)( ext_buffer );
    ASSERT_TRUE( !::NS(Buffer_is_in_developer_debug_mode)( ext_buffer ) );

    /* A call to NS(Buffer_refresh)() should fix the discrepancy between
     * ext_buffer and buffer */

    ASSERT_TRUE( ::NS(Buffer_refresh)( ext_buffer ) == 0 );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ext_buffer ) ==
                 ::NS(Buffer_get_data_begin_addr)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer ) ==
                 ::NS(Buffer_get_capacity)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer ) ==
                 ::NS(Buffer_get_size)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_objects)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_objects)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_slots)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_slots)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_dataptrs)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_dataptrs)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_garbage_ranges)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_garbage_ranges)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                 buffer->num_objects );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) ==
                 ::NS(Buffer_get_num_of_slots)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
                 ::NS(Buffer_get_num_of_dataptrs)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
                 ::NS(Buffer_get_num_of_garbage_ranges)( ext_buffer ) );

    /* --------------------------------------------------------------------- */
    /* Having filled up the buffer to its limit, it should not be possible to
     * add a another object instance as the buffer is *not* allowed to grow
     * -> test this now using the buffer */

    gen_obj_t* not_successfully_added = ::NS(GenericObj_add)( buffer,
            type_id_t{ 1 }, num_d_values, num_e_values,
            static_cast< i32_t >( TOTAL_NUM_OBJ ),
            static_cast< real_t >( TOTAL_NUM_OBJ ),
            &c_values[ 0 ], d_values.data(), e_values.data() );

    ASSERT_TRUE( not_successfully_added == nullptr );

    /* Nothing should have changed */

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ext_buffer ) ==
                 ::NS(Buffer_get_data_begin_addr)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer ) ==
                 ::NS(Buffer_get_capacity)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer ) ==
                 ::NS(Buffer_get_size)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_objects)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_objects)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_slots)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_slots)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_dataptrs)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_dataptrs)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_max_num_of_garbage_ranges)( ext_buffer ) ==
                 ::NS(Buffer_get_max_num_of_garbage_ranges)( buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( buffer ) ==
                 ::NS(Buffer_get_num_of_objects)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( buffer ) ==
                 ::NS(Buffer_get_num_of_slots)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( buffer ) ==
                 ::NS(Buffer_get_num_of_dataptrs)( ext_buffer ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( buffer ) ==
                 ::NS(Buffer_get_num_of_garbage_ranges)( ext_buffer ) );

    /* --------------------------------------------------------------------- */
    /* We delete the mapped buffer -> this should have no consequences on    *
     * ext_buffer as it is the ext_buffer who owns the data */

    ::NS(Buffer_delete)( buffer );
    buffer = nullptr;

    /* Verify that we can still access the data via ext_buffer: */

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( ext_buffer ) ==
        TOTAL_NUM_OBJ );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( ext_buffer ) >
        buf_size_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( ext_buffer ) ==
        ( ::NS(GenericObj_predict_required_num_dataptrs)(
            ext_buffer, num_d_values, num_e_values ) *
          ::NS(Buffer_get_num_of_objects)( ext_buffer ) ) );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( ext_buffer ) ==
        buf_size_t{ 0 } );

    ::NS(Object) const* it  =
        ::NS(Buffer_get_const_objects_begin)( ext_buffer );

    ::NS(Object) const* end  =
        ::NS(Buffer_get_const_objects_end)( ext_buffer );

    ASSERT_TRUE( ( it != nullptr ) && ( end != nullptr ) );
    ASSERT_TRUE( std::distance( it, end ) ==
                 static_cast< std::ptrdiff_t >( TOTAL_NUM_OBJ ) );

    for( buf_size_t ii = buf_size_t{ 0 } ; it != end ; ++it, ++ii )
    {
        d_values = saved_d_values[ ii ];
        e_values = saved_e_values[ ii ];

        SIXTRL_ASSERT( d_values.size() == num_d_values );
        SIXTRL_ASSERT( e_values.size() == num_e_values );

        ASSERT_TRUE( d_values.size() == num_d_values );
        ASSERT_TRUE( e_values.size() == num_e_values );

        ASSERT_TRUE( ::NS(Object_get_type_id)( it ) == type_id_t{ 1 } );
        ASSERT_TRUE( ::NS(Object_get_size)( it ) > sizeof( gen_obj_t ) );

        gen_obj_t const* obj = reinterpret_cast< gen_obj_t const* >(
            ::NS(Object_get_const_begin_ptr)( it ) );

        ASSERT_TRUE( obj != nullptr );
        ASSERT_TRUE( obj->a == static_cast< i32_t >( ii ) );
        ASSERT_TRUE( std::abs( obj->b - static_cast< real_t >( ii ) ) < EPS );
        ASSERT_TRUE( std::equal( &c_values[ 0 ], &c_values[ 4 ], obj->c ) );

        ASSERT_TRUE( obj->d != nullptr );
        ASSERT_TRUE( obj->num_d == num_d_values );
        ASSERT_TRUE( std::equal( d_values.begin(), d_values.end(), obj->d ) );

        ASSERT_TRUE( obj->e != nullptr );
        ASSERT_TRUE( obj->num_e == num_e_values );
        ASSERT_TRUE( std::equal( e_values.begin(), e_values.end(), obj->e ) );
    }

    /* --------------------------------------------------------------------- */
    /* Cleanup */

    ::NS(Buffer_delete)( ext_buffer );
    ext_buffer = nullptr;
}

/* end: tests/sixtracklib/common/test_buffer_c99.cpp */
