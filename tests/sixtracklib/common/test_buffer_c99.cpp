#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <limits>
#include <iterator>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/testlib/generic_buffer_obj.h"

#include "sixtracklib/common/buffer.h"

namespace sixtrl
{
    struct MyObj
    {
        st_object_type_id_t      type_id  SIXTRL_ALIGN( 8u );
        int32_t                  a        SIXTRL_ALIGN( 8u );
        double                   b        SIXTRL_ALIGN( 8u );
        double                   c[ 4 ]   SIXTRL_ALIGN( 8u );
        uint8_t* SIXTRL_RESTRICT d        SIXTRL_ALIGN( 8u );
        double*  SIXTRL_RESTRICT e        SIXTRL_ALIGN( 8u );
    };
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, InitOnExistingFlatMemory)
{
    std::vector< unsigned char > too_small( 36u, uint8_t{ 0 } );
    std::vector< unsigned char > data_buffer( ( 1u << 20u ), uint8_t{ 0 } );

    st_Buffer buffer;
    st_Buffer_preset( &buffer );

    int success = st_Buffer_init_on_flat_memory(
        &buffer, too_small.data(), too_small.size() );

    ASSERT_TRUE( success != 0 );

    st_Buffer_preset( &buffer );

    success = st_Buffer_init_on_flat_memory(
        &buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, InitFlatMemoryDataStoreAddObjectsRemapAndCompare )
{
    using my_obj_t = sixtrl::MyObj;
    using obj_t    = NS(Object);

    /* --------------------------------------------------------------------- */
    /* WRITING TO FLAT_MEMORY_BUFFER                                         */
    /* --------------------------------------------------------------------- */

    std::vector< unsigned char > data_buffer( 4096u, uint8_t{ 0 } );

    st_buffer_size_t const N = 3u;
    constexpr st_buffer_size_t NUM_DATAPTRS = 2u;

    st_buffer_size_t const offsets[ NUM_DATAPTRS ] =
    {
        offsetof( my_obj_t, d ),
        offsetof( my_obj_t, e )
    };

    st_buffer_size_t const type_sizes[ NUM_DATAPTRS ] =
    {
        sizeof( uint8_t ), sizeof( double )
    };

    st_buffer_size_t attr_counts[ NUM_DATAPTRS ] =
    {
        st_buffer_size_t{ 0 },
        st_buffer_size_t{ 0 }
    };

    st_Buffer buffer;
    st_Buffer_preset( &buffer );

    int success = st_Buffer_init_on_flat_memory_detailed( &buffer,
            data_buffer.data(), data_buffer.size(), N, 48u, 8u, 1u );

    ASSERT_TRUE( success == 0 );

    constexpr st_buffer_size_t num_d_values = 4;
    constexpr st_buffer_size_t num_e_values = 2;

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

    obj_t* ptr_object = st_Buffer_add_object( &buffer, &obj1, sizeof( obj1 ),
        obj1.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) == obj1.type_id );
    ASSERT_TRUE( st_Object_get_size( ptr_object ) > sizeof( obj1 ) );

    sixtrl::MyObj* ptr_stored_obj = reinterpret_cast< sixtrl::MyObj* >(
       st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj1 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj1.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj1.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj1.b ) <=
                 std::numeric_limits< double >::epsilon() );

    for( st_buffer_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj1.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] - obj1.c[ ii ] )
            <= std::numeric_limits< double >::epsilon() );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != &obj1.d[ ii ] );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  obj1.d[ ii ] );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != &obj1.e[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - obj1.e[ ii ] )
            <= std::numeric_limits< double >::epsilon() );
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

    ptr_object = st_Buffer_add_object( &buffer, &obj2, sizeof( obj2 ),
        obj2.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) == obj2.type_id );
    ASSERT_TRUE( st_Object_get_size( ptr_object ) > sizeof( obj2 ) );

    ptr_stored_obj = reinterpret_cast< sixtrl::MyObj* >(
        st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj1 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj2.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj2.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj2.b ) <=
                 std::numeric_limits< double >::epsilon() );

    for( st_buffer_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj2.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj2.c[ ii ] ) <= std::numeric_limits< double >::epsilon() );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != &obj2.d[ ii ] );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  obj2.d[ ii ] );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != &obj2.e[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - obj2.e[ ii ] ) <=
            std::numeric_limits< double >::epsilon() );
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

    ptr_object = st_Buffer_add_object( &buffer, &obj3, sizeof( obj3 ),
        obj3.type_id, NUM_DATAPTRS, &offsets[ 0 ],
            &type_sizes[ 0 ], &attr_counts[ 0 ] );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) == obj3.type_id );
    ASSERT_TRUE( st_Object_get_size( ptr_object ) > sizeof( obj3 ) );

    ptr_stored_obj = reinterpret_cast< sixtrl::MyObj* >(
        st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj3 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj3.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj3.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj3.b ) <=
                 std::numeric_limits< double >::epsilon() );

    for( st_buffer_size_t ii = 0u ; ii < 4u ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->c[ ii ] != &obj3.c[ ii ] );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj3.c[ ii ] ) <= std::numeric_limits< double >::epsilon() );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->d[ ii ] != nullptr );
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  uint8_t{ 0 } );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( &ptr_stored_obj->e[ ii ] != nullptr );
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - double{ 0.0 } )
            <= std::numeric_limits< double >::epsilon() );
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

    st_Buffer cmp_buffer;
    st_Buffer_preset( &cmp_buffer );

    success = st_Buffer_init_from_data(
        &cmp_buffer, copy_buffer.data(), copy_buffer.size() );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( st_Buffer_get_size( &buffer ) ==
                 st_Buffer_get_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_data_begin_addr( &cmp_buffer ) != 0u );
    ASSERT_TRUE( st_Buffer_get_data_end_addr(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( st_Buffer_get_data_begin_addr( &buffer ) !=
                 st_Buffer_get_data_begin_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_slot_size( &buffer ) ==
                 st_Buffer_get_slot_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_header_size( &buffer ) ==
                 st_Buffer_get_header_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_begin_addr( &cmp_buffer ) != 0u );
    ASSERT_TRUE( st_Buffer_get_objects_end_addr(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( st_Buffer_get_objects_begin_addr( &buffer ) !=
                 st_Buffer_get_objects_begin_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_end_addr( &buffer ) !=
                 st_Buffer_get_objects_end_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_objects( &buffer ) ==
                 st_Buffer_get_num_of_objects( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ASSERT_TRUE( st_Buffer_get_slots_size( &buffer ) ==
                 st_Buffer_get_slots_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_slots( &buffer ) ==
                 st_Buffer_get_num_of_slots( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_size( &buffer ) ==
                 st_Buffer_get_objects_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_dataptrs_size( &buffer ) ==
                 st_Buffer_get_dataptrs_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_dataptrs( &buffer ) ==
                 st_Buffer_get_num_of_dataptrs( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_garbage_size( &buffer ) ==
                 st_Buffer_get_garbage_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_garbage_ranges( &buffer ) ==
                 st_Buffer_get_num_of_garbage_ranges( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_t const* obj_it  = st_Buffer_get_const_objects_begin( &buffer );
    obj_t const* obj_end = st_Buffer_get_const_objects_end( &buffer );

    obj_t const* cmp_obj_it  = st_Buffer_get_const_objects_begin( &cmp_buffer );
    obj_t const* cmp_obj_end = st_Buffer_get_const_objects_end( &cmp_buffer );

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
        ASSERT_TRUE( st_Object_get_type_id( obj_it ) ==
                     st_Object_get_type_id( cmp_obj_it ) );

        ASSERT_TRUE( st_Object_get_size( obj_it ) ==
                     st_Object_get_size( cmp_obj_it ) );

        ASSERT_TRUE( st_Object_get_size( obj_it ) > sizeof( sixtrl::MyObj ) );

        ASSERT_TRUE( st_Object_get_const_begin_ptr( obj_it ) !=
                     st_Object_get_const_begin_ptr( cmp_obj_it ) );

        my_obj_t const* ptr_my_obj = reinterpret_cast< my_obj_t const* >(
            st_Object_get_const_begin_ptr( obj_it ) );

        my_obj_t const* ptr_cmp_obj = reinterpret_cast< my_obj_t const* >(
            st_Object_get_const_begin_ptr( cmp_obj_it ) );

        ASSERT_TRUE( ptr_my_obj  != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != ptr_my_obj );

        ASSERT_TRUE( ptr_cmp_obj->type_id == ptr_my_obj->type_id );
        ASSERT_TRUE( ptr_cmp_obj->a       == ptr_my_obj->a       );
        ASSERT_TRUE( std::fabs( ptr_cmp_obj->b - ptr_my_obj->b ) <=
                        std::numeric_limits< double >::epsilon() );

        for( st_buffer_size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( &ptr_cmp_obj->c[ ii ] != &ptr_my_obj->c[ ii ] );
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->c[ ii ] - ptr_my_obj->c[ ii ] )
                <= std::numeric_limits< double >::epsilon() );
        }

        ASSERT_TRUE( ptr_cmp_obj->d != ptr_my_obj->d );
        ASSERT_TRUE( ptr_cmp_obj->d != nullptr );
        ASSERT_TRUE( ptr_my_obj->d  != nullptr );

        for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
        {
            ASSERT_TRUE( ptr_cmp_obj->d[ ii ] == ptr_my_obj->d[ ii ] );
        }

        ASSERT_TRUE( ptr_cmp_obj->e != ptr_my_obj->e );
        ASSERT_TRUE( ptr_cmp_obj->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e  != nullptr );

        for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->e[ ii ] - ptr_my_obj->e[ ii ] )
                <= std::numeric_limits< double >::epsilon() );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Remap the same buffer again -> should work out just fine!             */
    /* --------------------------------------------------------------------- */

    st_Buffer_preset( &cmp_buffer );

    success = st_Buffer_init_from_data(
        &cmp_buffer, copy_buffer.data(), copy_buffer.size() );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( st_Buffer_get_size( &buffer ) ==
                 st_Buffer_get_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_data_begin_addr( &cmp_buffer ) != 0u );
    ASSERT_TRUE( st_Buffer_get_data_end_addr(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( st_Buffer_get_data_begin_addr( &buffer ) !=
                 st_Buffer_get_data_begin_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_slot_size( &buffer ) ==
                 st_Buffer_get_slot_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_header_size( &buffer ) ==
                 st_Buffer_get_header_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_begin_addr( &cmp_buffer ) != 0u );
    ASSERT_TRUE( st_Buffer_get_objects_end_addr(   &cmp_buffer ) != 0u );

    ASSERT_TRUE( st_Buffer_get_objects_begin_addr( &buffer ) !=
                 st_Buffer_get_objects_begin_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_end_addr( &buffer ) !=
                 st_Buffer_get_objects_end_addr( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_objects( &buffer ) ==
                 st_Buffer_get_num_of_objects( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ASSERT_TRUE( st_Buffer_get_slots_size( &buffer ) ==
                 st_Buffer_get_slots_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_slots( &buffer ) ==
                 st_Buffer_get_num_of_slots( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_objects_size( &buffer ) ==
                 st_Buffer_get_objects_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_dataptrs_size( &buffer ) ==
                 st_Buffer_get_dataptrs_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_dataptrs( &buffer ) ==
                 st_Buffer_get_num_of_dataptrs( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_garbage_size( &buffer ) ==
                 st_Buffer_get_garbage_size( &cmp_buffer ) );

    ASSERT_TRUE( st_Buffer_get_num_of_garbage_ranges( &buffer ) ==
                 st_Buffer_get_num_of_garbage_ranges( &cmp_buffer ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_it  = st_Buffer_get_const_objects_begin( &buffer );
    obj_end = st_Buffer_get_const_objects_end(   &buffer );

    cmp_obj_it  = st_Buffer_get_const_objects_begin( &cmp_buffer );
    cmp_obj_end = st_Buffer_get_const_objects_end(   &cmp_buffer );

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
        ASSERT_TRUE( st_Object_get_type_id( obj_it ) ==
                     st_Object_get_type_id( cmp_obj_it ) );

        ASSERT_TRUE( st_Object_get_size( obj_it ) ==
                     st_Object_get_size( cmp_obj_it ) );

        ASSERT_TRUE( st_Object_get_size( obj_it ) > sizeof( sixtrl::MyObj ) );

        ASSERT_TRUE( st_Object_get_const_begin_ptr( obj_it ) !=
                     st_Object_get_const_begin_ptr( cmp_obj_it ) );

        my_obj_t const* ptr_my_obj = reinterpret_cast< my_obj_t const* >(
            st_Object_get_const_begin_ptr( obj_it ) );

        my_obj_t const* ptr_cmp_obj = reinterpret_cast< my_obj_t const* >(
            st_Object_get_const_begin_ptr( cmp_obj_it ) );

        ASSERT_TRUE( ptr_my_obj  != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != nullptr );
        ASSERT_TRUE( ptr_cmp_obj != ptr_my_obj );

        ASSERT_TRUE( ptr_cmp_obj->type_id == ptr_my_obj->type_id );
        ASSERT_TRUE( ptr_cmp_obj->a       == ptr_my_obj->a       );
        ASSERT_TRUE( std::fabs( ptr_cmp_obj->b - ptr_my_obj->b ) <=
                        std::numeric_limits< double >::epsilon() );

        for( st_buffer_size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( &ptr_cmp_obj->c[ ii ] != &ptr_my_obj->c[ ii ] );
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->c[ ii ] - ptr_my_obj->c[ ii ] )
                <= std::numeric_limits< double >::epsilon() );
        }

        ASSERT_TRUE( ptr_cmp_obj->d != ptr_my_obj->d );
        ASSERT_TRUE( ptr_cmp_obj->d != nullptr );
        ASSERT_TRUE( ptr_my_obj->d  != nullptr );

        for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
        {
            ASSERT_TRUE( ptr_cmp_obj->d[ ii ] == ptr_my_obj->d[ ii ] );
        }

        ASSERT_TRUE( ptr_cmp_obj->e != ptr_my_obj->e );
        ASSERT_TRUE( ptr_cmp_obj->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e  != nullptr );

        for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ptr_cmp_obj->e[ ii ] - ptr_my_obj->e[ ii ] )
                <= std::numeric_limits< double >::epsilon() );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Free Buffer -> not necessary, but should also do no harm! */

    st_Buffer_free( &buffer );
    st_Buffer_free( &cmp_buffer );
}

/* ************************************************************************* */

TEST( C99_CommonBufferTests, ReconstructFromCObjectFile )
{
    using my_obj_t = sixtrl::MyObj;
    using obj_t    = NS(Object);

    /* --------------------------------------------------------------------- */
    /* PREPARE DATA OBJECTS                                                  */
    /* --------------------------------------------------------------------- */

    std::vector< unsigned char > data_buffer( 4096u, uint8_t{ 0 } );

    st_buffer_size_t const N = 3u;
    constexpr st_buffer_size_t NUM_DATAPTRS = 2u;

    st_buffer_size_t const offsets[ NUM_DATAPTRS ] =
    {
        offsetof( my_obj_t, d ),
        offsetof( my_obj_t, e )
    };

    st_buffer_size_t const type_sizes[ NUM_DATAPTRS ] =
    {
        sizeof( uint8_t ), sizeof( double )
    };

    st_buffer_size_t attr_counts[ NUM_DATAPTRS ] =
    {
        st_buffer_size_t{ 0 },
        st_buffer_size_t{ 0 }
    };

    constexpr st_buffer_size_t num_d_values = 4;
    constexpr st_buffer_size_t num_e_values = 2;

    uint8_t obj1_d_values[ num_d_values ] = { 31, 32, 33, 34 };
    double  obj1_e_values[ num_e_values ] = { 35.0, 36.0 };

    attr_counts[ 0 ] = num_d_values;
    attr_counts[ 1 ] = num_e_values;

    my_obj_t cmp_my_obj[ 3 ];
    std::memset( &cmp_my_obj[ 0 ], ( int )0, sizeof( my_obj_t ) * 3u );

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

        st_Buffer buffer;
        st_Buffer_preset( &buffer );

        int success = st_Buffer_init_on_flat_memory_detailed( &buffer,
                data_buffer.data(), data_buffer.size(), N, 48u, 8u, 1u );

        ASSERT_TRUE( success == 0 );

        obj_t* ptr_object = st_Buffer_add_object( &buffer, &obj1, sizeof( obj1 ),
            obj1.type_id, NUM_DATAPTRS, &offsets[ 0 ],
                &type_sizes[ 0 ], &attr_counts[ 0 ] );

        ASSERT_TRUE( ptr_object != nullptr );

        ptr_object = st_Buffer_add_object( &buffer, &obj2, sizeof( obj2 ),
            obj2.type_id, NUM_DATAPTRS, &offsets[ 0 ],
                &type_sizes[ 0 ], &attr_counts[ 0 ] );

        ASSERT_TRUE( ptr_object != nullptr );

        st_buffer_size_t const cnt = std::fwrite( ( unsigned char const* )(
            uintptr_t )st_Buffer_get_data_begin_addr( &buffer ),
            st_Buffer_get_size( &buffer ), st_buffer_size_t{ 1 }, fp );

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

    st_Buffer buffer;
    st_Buffer_preset( &buffer );

    int success = st_Buffer_init_from_data(
        &buffer, data_buffer_begin, buffer_size );

    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( st_Buffer_get_size( &buffer ) == buffer_size );
    ASSERT_TRUE( st_Buffer_get_num_of_objects( &buffer ) == 2u );

    st_Object const* obj_it  = st_Buffer_get_const_objects_begin( &buffer );
    st_Object const* obj_end = st_Buffer_get_const_objects_end( &buffer );

    sixtrl::MyObj const* cmp_obj_it = &cmp_my_obj[ 0 ];

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_obj_it )
    {
        using ptr_my_obj_t = sixtrl::MyObj const*;

        ASSERT_TRUE( st_Object_get_type_id( obj_it ) == cmp_obj_it->type_id );
        ASSERT_TRUE( st_Object_get_size( obj_it ) > sizeof( sixtrl::MyObj ) );
        ASSERT_TRUE( st_Object_get_const_begin_ptr( obj_it ) != nullptr );

        ptr_my_obj_t ptr_my_obj = reinterpret_cast< ptr_my_obj_t >(
                st_Object_get_const_begin_ptr( obj_it ) );

        ASSERT_TRUE( ptr_my_obj != nullptr );
        ASSERT_TRUE( ptr_my_obj->type_id == st_Object_get_type_id( obj_it ) );

        ASSERT_TRUE( ptr_my_obj->a == cmp_obj_it->a );
        ASSERT_TRUE( std::fabs( ( ptr_my_obj->b - cmp_obj_it->b ) <
                        std::numeric_limits< double >::epsilon() ) );

        for( std::size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ( ptr_my_obj->c[ ii ] -
                cmp_obj_it->c[ ii ] ) < std::numeric_limits<
                    double >::epsilon() ) );
        }

        ASSERT_TRUE( ptr_my_obj->d != nullptr );
        ASSERT_TRUE( cmp_my_obj->d != nullptr );

        for( std::size_t ii = 0u ; ii < 4u ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ( ptr_my_obj->d[ ii ] -
                cmp_obj_it->d[ ii ] ) < std::numeric_limits<
                    double >::epsilon() ) );
        }

        ASSERT_TRUE( ptr_my_obj->e != nullptr );
        ASSERT_TRUE( cmp_obj_it->e != nullptr );
        ASSERT_TRUE( ptr_my_obj->e[ 0 ] == cmp_obj_it->e[ 0 ] );
        ASSERT_TRUE( ptr_my_obj->e[ 1 ] == cmp_obj_it->e[ 1 ] );
    }

    st_Buffer_free( &buffer );
}

TEST( C99_CommonBufferTests, NewBufferAndGrowingWithinCapacity )
{
    using my_obj_t   = sixtrl::MyObj;
    using obj_t      = NS(Object);
    using buf_size_t = st_buffer_size_t;
    using type_id_t  = st_object_type_id_t;

    static constexpr buf_size_t ZERO_SIZE = buf_size_t{ 0u };

    st_Buffer* buffer = st_Buffer_new( buf_size_t{ 1u << 20u } );

    ASSERT_TRUE( buffer != nullptr );

    ASSERT_TRUE(  st_Buffer_has_datastore( buffer ) );
    ASSERT_TRUE(  st_Buffer_uses_datastore( buffer ) );
    ASSERT_TRUE(  st_Buffer_owns_datastore( buffer ) );
    ASSERT_TRUE(  st_Buffer_allow_modify_datastore_contents( buffer ) );
    ASSERT_TRUE(  st_Buffer_allow_clear( buffer ) );
    ASSERT_TRUE(  st_Buffer_allow_append_objects( buffer ) );
    ASSERT_TRUE(  st_Buffer_allow_remapping( buffer ) );
    ASSERT_TRUE(  st_Buffer_allow_resize( buffer ) );
    ASSERT_TRUE(  st_Buffer_uses_mempool_datastore( buffer ) );
    ASSERT_TRUE( !st_Buffer_uses_special_opencl_datastore( buffer ) );
    ASSERT_TRUE( !st_Buffer_uses_special_cuda_datastore( buffer ) );

    buf_size_t const slot_size = st_Buffer_get_slot_size( buffer );
    ASSERT_TRUE( slot_size > ZERO_SIZE );

    buf_size_t const sect_hdr_len = st_Buffer_get_section_header_size( buffer );

    ASSERT_TRUE( st_Buffer_get_size( buffer ) > ZERO_SIZE );

    ASSERT_TRUE( st_Buffer_get_num_of_slots( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_max_num_of_slots( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_slots_size( buffer ) == sect_hdr_len );

    ASSERT_TRUE( st_Buffer_get_num_of_objects( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_max_num_of_objects( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_objects_size( buffer ) == sect_hdr_len );

    ASSERT_TRUE( st_Buffer_get_num_of_dataptrs( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_max_num_of_dataptrs( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_dataptrs_size( buffer ) == sect_hdr_len );

    ASSERT_TRUE( st_Buffer_get_num_of_garbage_ranges( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_max_num_of_garbage_ranges( buffer ) == ZERO_SIZE );
    ASSERT_TRUE( st_Buffer_get_garbage_size( buffer ) == sect_hdr_len );

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
    std::memset( &obj1, ( int )0, sizeof( my_obj_t ) );

    obj1.type_id   = 3u;
    obj1.a         = 1;
    obj1.b         = 2.0;
    obj1.c[ 0 ]    = 3.0;
    obj1.c[ 1 ]    = 4.0;
    obj1.c[ 2 ]    = 5.0;
    obj1.c[ 3 ]    = 6.0;
    obj1.d         = nullptr;
    obj1.e         = nullptr;

    obj_t* ptr_object = st_Buffer_add_object( buffer, &obj1, sizeof( my_obj_t ),
        static_cast< type_id_t >( obj1.type_id ), num_dataptrs, attr_offsets,
            attr_sizes, attr_counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Buffer_get_num_of_objects( buffer ) == buf_size_t{ 1u } );
    ASSERT_TRUE( st_Buffer_get_num_of_slots( buffer ) > ZERO_SIZE );

    ASSERT_TRUE( st_Object_get_size( ptr_object ) >= sizeof( my_obj_t ) );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) ==
                 static_cast< type_id_t >( obj1.type_id ) );

    ASSERT_TRUE( st_Object_get_begin_ptr( ptr_object ) != nullptr );

    my_obj_t* ptr_my_obj = reinterpret_cast< my_obj_t* >(
        st_Object_get_begin_ptr( ptr_object ) );

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
        st_ManagedBuffer_get_slot_based_length(
            num_d_values * sizeof( uint8_t ), slot_size ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
     * Add second obj -> obj2
     * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    size_t const num_objects_after_obj1 =
        st_Buffer_get_num_of_objects( buffer );

    size_t const num_slots_after_obj1    =
        st_Buffer_get_num_of_slots( buffer );

    size_t const num_dataptrs_after_obj1 =
        st_Buffer_get_num_of_dataptrs( buffer );

    size_t const num_garbage_ranges_after_obj1 =
        st_Buffer_get_num_of_garbage_ranges( buffer );

    my_obj_t obj2;
    std::memset( &obj2, ( int )0, sizeof( my_obj_t ) );

    obj2.type_id   = 9u;
    obj2.a         = 10;
    obj2.b         = 12.0;
    obj2.c[ 0 ]    = 13.0;
    obj2.c[ 1 ]    = 14.0;
    obj2.c[ 2 ]    = 15.0;
    obj2.c[ 3 ]    = 16.0;
    obj2.d         = nullptr;
    obj2.e         = nullptr;

    ptr_object = st_Buffer_add_object( buffer, &obj2, sizeof( my_obj_t ),
        static_cast< type_id_t >( obj2.type_id ), num_dataptrs, attr_offsets,
            attr_sizes, attr_counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Buffer_get_num_of_objects( buffer ) ==
        ( num_objects_after_obj1 + 1u ) );

    ASSERT_TRUE( st_Buffer_get_num_of_slots( buffer ) >
        ( num_slots_after_obj1 ) );

    ASSERT_TRUE( st_Buffer_get_num_of_dataptrs( buffer ) ==
        ( num_dataptrs_after_obj1 + num_dataptrs ) );

    ASSERT_TRUE( st_Buffer_get_num_of_garbage_ranges( buffer ) ==
                 num_garbage_ranges_after_obj1 );

    ASSERT_TRUE( st_Object_get_size( ptr_object ) >= sizeof( my_obj_t ) );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) ==
                 static_cast< type_id_t >( obj2.type_id ) );

    ASSERT_TRUE( st_Object_get_begin_ptr( ptr_object ) != nullptr );

    unsigned char const* ptr_obj1_e =
        reinterpret_cast< unsigned char const* >( ptr_my_obj->e );

    ptr_my_obj = reinterpret_cast< my_obj_t* >(
        st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_my_obj != nullptr );

    std::ptrdiff_t dist_obj1_e_to_obj2 = std::distance( ptr_obj1_e,
        reinterpret_cast< unsigned char const* >( ptr_my_obj ) );

    ASSERT_TRUE( dist_obj1_e_to_obj2 > 0 );
    ASSERT_TRUE( static_cast< size_t >( dist_obj1_e_to_obj2 ) >=
        st_ManagedBuffer_get_slot_based_length(
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
        st_ManagedBuffer_get_slot_based_length(
            num_d_values * sizeof( uint8_t ), slot_size ) );

    /* --------------------------------------------------------------------- *
     * CLEANUP & RESSOURCE MANAGEMENT                                        *
     * --------------------------------------------------------------------- */

    st_Buffer_delete( buffer );
    buffer = nullptr;
}

TEST( C99_CommonBufferTests, AddGenericObjectsTestAutoGrowingOfBuffer )
{
    using buf_size_t    = ::st_buffer_size_t;
    using type_id_t     = ::st_object_type_id_t;
    using generic_obj_t = ::st_GenericObj;

    ::st_Buffer* buffer = ::st_Buffer_new( buf_size_t{ 0 } );

    ASSERT_TRUE( ::st_Buffer_allow_resize( buffer ) );
    ASSERT_TRUE( ::st_Buffer_allow_remapping( buffer ) );
    ASSERT_TRUE( ::st_Buffer_allow_append_objects( buffer ) );

    buf_size_t prev_capacity     = ::st_Buffer_get_capacity( buffer );
    buf_size_t prev_size         = ::st_Buffer_get_size( buffer );
    buf_size_t prev_num_objects  = ::st_Buffer_get_num_of_objects( buffer );
    buf_size_t prev_num_slots    = ::st_Buffer_get_num_of_slots( buffer );
    buf_size_t prev_num_dataptrs = ::st_Buffer_get_num_of_dataptrs( buffer );

    ASSERT_TRUE( buffer            != nullptr );
    ASSERT_TRUE( prev_size         >  buf_size_t{ 0 } );
    ASSERT_TRUE( prev_capacity     >= prev_size );
    ASSERT_TRUE( prev_num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_dataptrs == buf_size_t{ 0 } );

    constexpr buf_size_t NUM_OBJECTS_TO_ADD = 100;
    constexpr buf_size_t num_d_values = 10;
    constexpr buf_size_t num_e_values = 10;

    buf_size_t ii = 0;

    for( ; ii < NUM_OBJECTS_TO_ADD ; ++ii )
    {
        generic_obj_t* obj = ::st_GenericObj_new(
            buffer, static_cast< type_id_t >( ii ), num_d_values, num_e_values );

        ASSERT_TRUE( obj != nullptr );

        buf_size_t const capacity = ::st_Buffer_get_capacity( buffer );
        buf_size_t const size     = ::st_Buffer_get_size( buffer );

        ASSERT_TRUE( capacity >= prev_capacity );
        ASSERT_TRUE( size     >  prev_size     );
        ASSERT_TRUE( capacity >= size  );

        prev_capacity = capacity;
        prev_size     = size;

        buf_size_t const num_objects  = ::st_Buffer_get_num_of_objects( buffer );
        buf_size_t const num_slots    = ::st_Buffer_get_num_of_slots( buffer );
        buf_size_t const num_dataptrs = ::st_Buffer_get_num_of_dataptrs( buffer );

        ASSERT_TRUE( num_objects  == prev_num_objects + buf_size_t{ 1 } );
        ASSERT_TRUE( num_slots    >= prev_num_slots );
        ASSERT_TRUE( num_dataptrs >= prev_num_dataptrs );

        prev_num_objects  = num_objects;
        prev_num_slots    = num_slots;
        prev_num_dataptrs = num_dataptrs;
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( buffer ) == NUM_OBJECTS_TO_ADD );

    ::st_Buffer_delete( buffer );
}

/* end: tests/sixtracklib/common/test_buffer_c99.cpp */
