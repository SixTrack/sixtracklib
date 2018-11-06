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
#include "sixtracklib/testlib/common/generic_buffer_obj.h"
#include "sixtracklib/testlib/testdata/testdata_files.h"
#include "sixtracklib/common/buffer.hpp"

#include "sixtracklib/testlib.h"

namespace sixtrack
{
    struct MyObj
    {
        st_object_type_id_t      type_id  SIXTRL_ALIGN( 8u );
        int32_t                  a        SIXTRL_ALIGN( 8u );
        double                   b        SIXTRL_ALIGN( 8u );
        double                   c[ 4 ]   SIXTRL_ALIGN( 8u );
        uint8_t* SIXTRL_RESTRICT d        SIXTRL_ALIGN( 8u );
        double*  SIXTRL_RESTRICT e        SIXTRL_ALIGN( 8u );

        void preset( st_object_type_id_t const type_id =
            st_object_type_id_t{ 0 } ) SIXTRL_NOEXCEPT
        {
            this->type_id = type_id;
            this->a       = int32_t{ 0 };
            this->b       = double{ 0.0 };
            this->d       = nullptr;
            this->e       = nullptr;

            std::fill( &this->c[ 0 ], &this->c[ 4 ], double{ 0.0 } );

            return;
        }

        SIXTRL_FN static bool CanAddToBuffer(
            ::NS(Buffer)& SIXTRL_RESTRICT_REF buffer,
            sixtrack::Buffer::size_type const num_d_values,
            sixtrack::Buffer::size_type const num_e_values,
            sixtrack::Buffer::size_type* SIXTRL_RESTRICT ptr_requ_objects  = nullptr,
            sixtrack::Buffer::size_type* SIXTRL_RESTRICT ptr_requ_slots    = nullptr,
            sixtrack::Buffer::size_type* SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT
        {
            using _this_t   = MyObj;
            using  buffer_t = sixtrack::Buffer;
            using  size_t   = buffer_t::size_type;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            size_t const d_size       = sizeof( uint8_t );
            size_t const e_size       = sizeof( double );
            size_t const num_dataptrs = size_t{ 2u };

            size_t const sizes[]  = { d_size, e_size };
            size_t const counts[] = { num_d_values, num_e_values };

            return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ),
                    num_dataptrs, sizes, counts, ptr_requ_objects,
                        ptr_requ_slots, ptr_requ_dataptrs );
        }

        SIXTRL_FN static MyObj* CreateNewOnBuffer(
            ::NS(Buffer)& SIXTRL_RESTRICT_REF buffer,
            sixtrack::Buffer::type_id_t const type_id,
            sixtrack::Buffer::size_type const num_d_values,
            sixtrack::Buffer::size_type const num_e_values )
        {
            using _this_t   = sixtrack::MyObj;
            using  buffer_t = sixtrack::Buffer;
            using  size_t   = buffer_t::size_type;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            size_t const d_size       = sizeof( uint8_t );
            size_t const e_size       = sizeof( double );
            size_t const num_dataptrs = size_t{ 2u };

            size_t const offsets[] =
            {
                offsetof( _this_t, d ),
                offsetof( _this_t, e )
            };

            size_t const sizes[]   = { d_size, e_size };
            size_t const counts[]  = { num_d_values, num_e_values };

            _this_t obj;

            /* Only safe because is_trivial and is_standard_layout is true!! */
            std::memset( &obj, ( int )0, sizeof( obj ) );

            obj.preset( type_id );

            buffer_t::object_t* ptr_object = ::NS(Buffer_add_object)(
                &buffer, &obj, sizeof( obj ), type_id,
                    num_dataptrs, offsets, sizes, counts );

            return reinterpret_cast< sixtrack::MyObj* >(
                ::NS(Object_get_begin_ptr)( ptr_object ) );
        }

        SIXTRL_FN static MyObj* AddToBuffer(
            ::NS(Buffer)& SIXTRL_RESTRICT_REF buffer,
            sixtrack::Buffer::type_id_t const type_id,
            sixtrack::Buffer::size_type const num_d_values,
            sixtrack::Buffer::size_type const num_e_values,
            int32_t const  a_value = int32_t{ 0 },
            double  const  b_value = double{ 0.0 },
            double  const* SIXTRL_RESTRICT c_ptr = nullptr,
            uint8_t*       SIXTRL_RESTRICT d_ptr = nullptr,
            double*        SIXTRL_RESTRICT e_ptr = nullptr  )
        {
            using _this_t   = sixtrack::MyObj;
            using  buffer_t = sixtrack::Buffer;
            using  size_t   = buffer_t::size_type;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            size_t const d_size       = sizeof( uint8_t );
            size_t const e_size       = sizeof( double );
            size_t const num_dataptrs = size_t{ 2u };

            size_t const offsets[] =
            {
                offsetof( _this_t, d ),
                offsetof( _this_t, e )
            };

            size_t const sizes[]   = { d_size, e_size };
            size_t const counts[]  = { num_d_values, num_e_values };

            _this_t obj;

            /* Only safe because is_trivial and is_standard_layout is true!! */
            std::memset( &obj, ( int )0, sizeof( obj ) );

            obj.type_id = type_id;
            obj.a       = a_value;
            obj.b       = b_value;
            obj.d       = d_ptr;
            obj.e       = e_ptr;

            if( c_ptr != nullptr )
            {
                std::copy( c_ptr, c_ptr + 4, &obj.c[ 0 ] );
            }

            buffer_t::object_t* ptr_object = ::NS(Buffer_add_object)(
                &buffer, &obj, sizeof( obj ), type_id,
                    num_dataptrs, offsets, sizes, counts );

            return reinterpret_cast< MyObj* >(
                ::NS(Object_get_begin_ptr)( ptr_object ) );
        }
    };
}

/* ************************************************************************* */

TEST( CXX_CommonBufferTests, InitOnExistingFlatMemory)
{
    namespace st = sixtrack;

    using buffer_t = st::Buffer;
    using size_t   = buffer_t::size_type;

    std::vector< unsigned char > too_small( 36u, uint8_t{ 0 } );
    std::vector< unsigned char > data_buffer( ( 1u << 20u ), uint8_t{ 0 } );

    st::Buffer too_small_buffer( too_small.data(), too_small.size() );

    ASSERT_TRUE( too_small_buffer.size() == size_t{ 0 } );

    st::Buffer buffer( data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( buffer.capacity() == data_buffer.size() );
    ASSERT_TRUE( buffer.size() > size_t{ 0 } );
    ASSERT_TRUE( buffer.dataBegin< unsigned char const* >() ==
                 data_buffer.data() );
}

/* ************************************************************************* */

TEST( CXX_CommonBufferTests, InitFlatMemoryDataStoreAddObjectsRemapAndCompare )
{
    namespace st = sixtrack;

    using my_obj_t  = st::MyObj;
    using type_id_t = ::st_object_type_id_t;

    using buffer_t  = st::Buffer;
    using obj_t     = buffer_t::object_t;
    using size_t    = buffer_t::size_type;
    using raw_t     = unsigned char;

    /* --------------------------------------------------------------------- */
    /* WRITING TO FLAT_MEMORY_BUFFER                                         */
    /* --------------------------------------------------------------------- */

    constexpr size_t NUM_DATAPTRS_PER_OBJ = size_t{ 2u };

    size_t const u8_size  = sizeof( uint8_t );
    size_t const f64_size = sizeof( double  );

    size_t const d_offset = offsetof( st::MyObj, d );
    size_t const e_offset = offsetof( st::MyObj, e );

    size_t offsets[     NUM_DATAPTRS_PER_OBJ ] = { d_offset, e_offset };
    size_t type_sizes[  NUM_DATAPTRS_PER_OBJ ] = { u8_size, f64_size };
    size_t attr_counts[ NUM_DATAPTRS_PER_OBJ ] = { size_t{ 0 }, size_t{ 0 } };

    constexpr size_t NUM_SLOTS          = size_t{ 48u };
    constexpr size_t NUM_OBJECTS        = size_t{ 3u };
    constexpr size_t NUM_DATAPTRS       = NUM_OBJECTS * NUM_DATAPTRS_PER_OBJ;
    constexpr size_t NUM_GARBAGE_RANGES = size_t{ 1u };

    size_t const buffer_capacity = st::Buffer::CalculateBufferSize(
        NUM_OBJECTS, NUM_SLOTS, NUM_DATAPTRS, NUM_GARBAGE_RANGES );

    ASSERT_TRUE( buffer_capacity > size_t{ 0 } );

    std::vector< raw_t > data_buffer( buffer_capacity, raw_t{ 0 } );

    st::Buffer buffer( data_buffer.data(), NUM_OBJECTS, NUM_SLOTS,
                       NUM_DATAPTRS, NUM_GARBAGE_RANGES, data_buffer.size() );

    ASSERT_TRUE( buffer.size()     >= buffer_capacity );
    ASSERT_TRUE( buffer.capacity() >= buffer_capacity );
    ASSERT_TRUE( buffer.dataBegin< raw_t const* >() == data_buffer.data() );
    ASSERT_TRUE( buffer.dataEnd< raw_t const* >()   != nullptr );

    ASSERT_TRUE( buffer.getMaxNumObjects()  == NUM_OBJECTS );
    ASSERT_TRUE( buffer.getNumObjects()     == size_t{ 0 } );

    ASSERT_TRUE( buffer.getMaxNumSlots()    == NUM_SLOTS );
    ASSERT_TRUE( buffer.getNumSlots()       == size_t{ 0 } );

    ASSERT_TRUE( buffer.getMaxNumDataptrs() == NUM_DATAPTRS );
    ASSERT_TRUE( buffer.getNumDataptrs()    == size_t{ 0 } );

    /* --------------------------------------------------------------------- */

    constexpr size_t num_c_values = size_t{ 4 };
    constexpr size_t num_d_values = size_t{ 4 };
    constexpr size_t num_e_values = size_t{ 2 };

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

    obj_t* ptr_object = buffer.addObject( obj1, obj1.type_id,
        NUM_DATAPTRS_PER_OBJ, offsets, type_sizes, attr_counts );


    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( st_Object_get_type_id( ptr_object ) == obj1.type_id );
    ASSERT_TRUE( st_Object_get_size( ptr_object ) > sizeof( obj1 ) );

    sixtrack::MyObj* ptr_stored_obj = reinterpret_cast< sixtrack::MyObj* >(
       st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj != &obj1 );

    ASSERT_TRUE( ptr_stored_obj->type_id == obj1.type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj1.a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj1.b ) <=
                 std::numeric_limits< double >::epsilon() );

    for( st_buffer_size_t ii = 0u ; ii < num_c_values ; ++ii )
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

    type_id_t const obj2_type_id   = type_id_t{ 4u };
    int32_t   const obj2_a         = int32_t{ 100 };
    double    const obj2_b         = double{ 101.0 };
    double    const obj2_c[]       = { 102.0, 103.0, 104.0, 105.0 };

    uint8_t obj2_d[ num_d_values ] =
    {
        uint8_t{ 106 }, uint8_t{ 107 }, uint8_t{ 108 }, uint8_t{ 109 }
    };

    double obj2_e[ num_e_values ]  =
    {
        double{ 110.0 }, double{ 111.0 }
    };

    ptr_stored_obj = buffer.add< my_obj_t >( obj2_type_id,
        num_d_values, num_e_values, obj2_a, obj2_b, obj2_c, obj2_d, obj2_e );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj->type_id == obj2_type_id );
    ASSERT_TRUE( ptr_stored_obj->a       == obj2_a       );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - obj2_b ) <=
                 std::numeric_limits< double >::epsilon() );

    ASSERT_TRUE( &ptr_stored_obj->c[ 0 ] != &obj2_c[ 0 ] );

    for( size_t ii = 0u ; ii < num_c_values ; ++ii )
    {
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            obj2_c[ ii ] ) <= std::numeric_limits< double >::epsilon() );
    }

    ASSERT_TRUE( &ptr_stored_obj->d[ 0 ] != &obj2_d[ 0 ] );

    for( size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  obj2_d[ ii ] );
    }

    ASSERT_TRUE( &ptr_stored_obj->e[ 0 ] != &obj2_e[ 0 ] );

    for( size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - obj2_e[ ii ] ) <=
            std::numeric_limits< double >::epsilon() );
    }

    /* --------------------------------------------------------------------- */

    type_id_t const cmp_obj_id = type_id_t{ 5 };

    my_obj_t cmp_obj;
    cmp_obj.preset( cmp_obj_id );

    ptr_stored_obj = buffer.createNew< my_obj_t >(
        cmp_obj_id, num_d_values, num_e_values );

    ASSERT_TRUE( ptr_stored_obj != nullptr );
    ASSERT_TRUE( ptr_stored_obj->type_id == cmp_obj_id );
    ASSERT_TRUE( ptr_stored_obj->a       == cmp_obj.a  );
    ASSERT_TRUE( std::fabs( ptr_stored_obj->b - cmp_obj.b ) <=
                 std::numeric_limits< double >::epsilon() );

    ASSERT_TRUE( &ptr_stored_obj->c[ 0 ] != &cmp_obj.c[ 0 ] );

    for( st_buffer_size_t ii = 0u ; ii < num_c_values ; ++ii )
    {
        ASSERT_TRUE( std::fabs( ptr_stored_obj->c[ ii ] -
            cmp_obj.c[ ii ] ) <= std::numeric_limits< double >::epsilon() );
    }

    ASSERT_TRUE( ptr_stored_obj->d != nullptr );
    ASSERT_TRUE( ptr_stored_obj->e != nullptr );

    for( st_buffer_size_t ii = 0 ; ii < num_d_values ; ++ii )
    {
        ASSERT_TRUE(  ptr_stored_obj->d[ ii ] ==  uint8_t{ 0 } );
    }

    for( st_buffer_size_t ii = 0 ; ii < num_e_values ; ++ii )
    {
        ASSERT_TRUE( std::fabs( ptr_stored_obj->e[ ii ] - double{ 0.0 } )
            <= std::numeric_limits< double >::epsilon() );
    }

    /* --------------------------------------------------------------------- */
    /* COPYING FLAT MEMORY BUFFER TO DIFFERENT MEMORY BUFFER ->
     * THIS SIMULATES THE EFFECTS OF MOVING THE BUFFER TO A DIFFERENT
     * MEMORY REALM, LIKE FOR EXAMPLE ON A GPU-TYPE DEVICE (OR READING
     * FROM A FILE, ETC. )*/
    /* --------------------------------------------------------------------- */

    std::vector< raw_t > copy_buffer( buffer.capacity(), raw_t{ 0 } );

    copy_buffer.assign( buffer.dataBegin< raw_t const* >(),
                        buffer.dataEnd< raw_t const* >() );

    st::Buffer cmp_buffer( copy_buffer.data(), copy_buffer.size() );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    using address_t = buffer_t::address_t;

    ASSERT_TRUE( cmp_buffer.size() == buffer.size() );

    ASSERT_TRUE( cmp_buffer.getDataBeginAddr() != address_t{ 0u } );
    ASSERT_TRUE( cmp_buffer.getDataEndAddr()   != address_t{ 0u } );

    ASSERT_TRUE( buffer.getSlotSize()   == cmp_buffer.getSlotSize() );
    ASSERT_TRUE( buffer.getHeaderSize() == cmp_buffer.getHeaderSize() );

    ASSERT_TRUE( buffer.getSectionHeaderSize() ==
                 cmp_buffer.getSectionHeaderSize() );

    ASSERT_TRUE( buffer.getDataBeginAddr()  != cmp_buffer.getDataEndAddr() );
    ASSERT_TRUE( buffer.getDataEndAddr()    != cmp_buffer.getDataEndAddr() );

    ASSERT_TRUE( buffer.getIndexBeginAddr()     != address_t{ 0u } );
    ASSERT_TRUE( buffer.getIndexEndAddr()       != address_t{ 0u } );

    ASSERT_TRUE( cmp_buffer.getIndexBeginAddr() != address_t{ 0u } );
    ASSERT_TRUE( cmp_buffer.getIndexEndAddr()   != address_t{ 0u } );

    ASSERT_TRUE( cmp_buffer.getIndexBeginAddr() != buffer.getIndexBeginAddr() );
    ASSERT_TRUE( cmp_buffer.getIndexEndAddr()   != buffer.getIndexEndAddr() );

    ASSERT_TRUE( cmp_buffer.getNumObjects()     == buffer.getNumObjects() );
    ASSERT_TRUE( cmp_buffer.getMaxNumObjects()  == buffer.getMaxNumObjects() );
    ASSERT_TRUE( cmp_buffer.getObjectsSize()    == buffer.getObjectsSize() );

    ASSERT_TRUE( cmp_buffer.getNumSlots()       == buffer.getNumSlots() );
    ASSERT_TRUE( cmp_buffer.getMaxNumSlots()    == buffer.getMaxNumSlots() );
    ASSERT_TRUE( cmp_buffer.getSlotsSize()      == buffer.getSlotsSize() );

    ASSERT_TRUE( cmp_buffer.getNumDataptrs()    == buffer.getNumDataptrs() );
    ASSERT_TRUE( cmp_buffer.getMaxNumDataptrs() == buffer.getMaxNumDataptrs() );
    ASSERT_TRUE( cmp_buffer.getDataptrsSize()   == buffer.getDataptrsSize() );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    obj_t const* obj_it  = buffer.indexBegin< obj_t const* >();
    obj_t const* obj_end = buffer.indexEnd< obj_t const* >();

    obj_t const* cmp_obj_it  = cmp_buffer.indexBegin< obj_t const* >();
    obj_t const* cmp_obj_end = cmp_buffer.indexEnd< obj_t const* >();

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

        ASSERT_TRUE( st_Object_get_size( obj_it ) > sizeof( st::MyObj ) );

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

        for( st_buffer_size_t ii = 0u ; ii < num_c_values ; ++ii )
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

    st::Buffer cmp2_buffer( copy_buffer.data(), copy_buffer.size() );

    /* --------------------------------------------------------------------- */
    /* COMPARE THE OBJECTS ON THE TWO BUFFERS                                */
    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( cmp2_buffer.size() == cmp_buffer.size() );

    ASSERT_TRUE( cmp2_buffer.getDataBeginAddr() != address_t{ 0u } );
    ASSERT_TRUE( cmp2_buffer.getDataEndAddr()   != address_t{ 0u } );

    ASSERT_TRUE( cmp2_buffer.getSlotSize()   == cmp_buffer.getSlotSize() );
    ASSERT_TRUE( cmp2_buffer.getHeaderSize() == cmp_buffer.getHeaderSize() );

    ASSERT_TRUE( cmp2_buffer.getSectionHeaderSize() ==
                 cmp_buffer.getSectionHeaderSize() );

    ASSERT_TRUE( cmp2_buffer.getDataBeginAddr() == cmp_buffer.getDataBeginAddr() );
    ASSERT_TRUE( cmp2_buffer.getDataEndAddr()   == cmp_buffer.getDataEndAddr() );

    ASSERT_TRUE( cmp2_buffer.getIndexBeginAddr() != address_t{ 0u } );
    ASSERT_TRUE( cmp2_buffer.getIndexEndAddr()   != address_t{ 0u } );

    ASSERT_TRUE( cmp2_buffer.getIndexBeginAddr() ==
                 cmp_buffer.getIndexBeginAddr() );

    ASSERT_TRUE( cmp2_buffer.getIndexEndAddr()   ==
                 cmp_buffer.getIndexEndAddr() );

    ASSERT_TRUE( cmp2_buffer.getNumObjects()     == cmp_buffer.getNumObjects() );
    ASSERT_TRUE( cmp2_buffer.getMaxNumObjects()  == cmp_buffer.getMaxNumObjects() );
    ASSERT_TRUE( cmp2_buffer.getObjectsSize()    == cmp_buffer.getObjectsSize() );

    ASSERT_TRUE( cmp2_buffer.getNumSlots()       == cmp_buffer.getNumSlots() );
    ASSERT_TRUE( cmp2_buffer.getMaxNumSlots()    == cmp_buffer.getMaxNumSlots() );
    ASSERT_TRUE( cmp2_buffer.getSlotsSize()      == cmp_buffer.getSlotsSize() );

    ASSERT_TRUE( cmp2_buffer.getNumDataptrs()    == cmp_buffer.getNumDataptrs() );
    ASSERT_TRUE( cmp2_buffer.getMaxNumDataptrs() == cmp_buffer.getMaxNumDataptrs() );
    ASSERT_TRUE( cmp2_buffer.getDataptrsSize()   == cmp_buffer.getDataptrsSize() );

    obj_it  = cmp_buffer.indexBegin< obj_t const* >();
    obj_end = cmp_buffer.indexEnd< obj_t const* >();

    cmp_obj_it  = cmp2_buffer.indexBegin< obj_t const* >();
    cmp_obj_end = cmp2_buffer.indexEnd< obj_t const* >();

    ASSERT_TRUE( obj_it      != nullptr     );
    ASSERT_TRUE( obj_end     != nullptr     );
    ASSERT_TRUE( cmp_obj_it  != nullptr     );
    ASSERT_TRUE( cmp_obj_end != nullptr     );
    ASSERT_TRUE( obj_it      == cmp_obj_it  );
    ASSERT_TRUE( obj_end     == cmp_obj_end );
}

/* ************************************************************************* */

TEST( CXX_CommonBufferTests, ReconstructFromCObjectFile )
{
    namespace st = sixtrack;

    using my_obj_t  = st::MyObj;
    using type_id_t = ::st_object_type_id_t;

    using buffer_t  = st::Buffer;
    using obj_t     = buffer_t::object_t;
    using size_t    = buffer_t::size_type;
    using raw_t     = unsigned char;

    constexpr size_t num_c_values = size_t{ 4 };
    constexpr size_t num_d_values = size_t{ 4 };
    constexpr size_t num_e_values = size_t{ 2 };

    type_id_t const obj1_type_id = type_id_t{ 3 };
    int32_t const   obj1_a       = int32_t{ 25 };
    double  const   obj1_b       = double{ 26.0 };
    double  const   obj1_c[]     = { 26.0, 27.0, 28.0, 29.0 };
    uint8_t         obj1_d[ num_d_values ] = { 31, 32, 33, 34 };
    double          obj1_e[ num_e_values ] = { 35.0, 36.0 };

    type_id_t const obj2_type_id = type_id_t{ 4 };
    int32_t const   obj2_a       = int32_t{ 100 };
    double  const   obj2_b       = double{ 101.0 };
    double  const   obj2_c[]     = { 102.0, 103.0, 104.0, 105.0 };
    uint8_t         obj2_d[ num_d_values ] = { 106, 107, 108, 109 };
    double          obj2_e[ num_e_values ] = { 110.0, 111.0 };

    type_id_t const obj3_type_id = type_id_t{ 5 };
    my_obj_t obj3;
    obj3.preset( obj3_type_id );

    /* --------------------------------------------------------------------- */

    char const PATH_TO_BINARY_FILE[] = "./test.np";
    FILE* fp = std::fopen( PATH_TO_BINARY_FILE, "rb" );

    if( fp != nullptr )
    {
        std::fclose( fp );
        fp = nullptr;

        std::remove( PATH_TO_BINARY_FILE );

        fp = std::fopen( PATH_TO_BINARY_FILE, "rb" );
        ASSERT_TRUE( fp == nullptr );
    }

    if( fp == nullptr )
    {
        fp = std::fopen( PATH_TO_BINARY_FILE, "wb" );
        ASSERT_TRUE( fp != nullptr );

        constexpr size_t NUM_DATAPTRS_PER_OBJ = size_t{ 2u };
        constexpr size_t NUM_SLOTS            = size_t{ 48u };
        constexpr size_t NUM_OBJECTS          = size_t{ 3u };
        constexpr size_t NUM_DATAPTRS         = NUM_OBJECTS * NUM_DATAPTRS_PER_OBJ;
        constexpr size_t NUM_GARBAGE_RANGES   = size_t{ 1u };

        size_t const buffer_capacity = st::Buffer::CalculateBufferSize(
            NUM_OBJECTS, NUM_SLOTS, NUM_DATAPTRS, NUM_GARBAGE_RANGES );

        std::vector< raw_t > data_buffer( buffer_capacity, raw_t{ 0 } );

        st::Buffer buffer( data_buffer.data(), NUM_OBJECTS, NUM_SLOTS,
                        NUM_DATAPTRS, NUM_GARBAGE_RANGES, data_buffer.size() );

        /* --------------------------------------------------------------------- */

        my_obj_t* ptr_obj1 = buffer.add< my_obj_t >( obj1_type_id,
            num_d_values, num_e_values, obj1_a, obj1_b, obj1_c, obj1_d, obj1_e );

        ASSERT_TRUE( ptr_obj1 != SIXTRL_NULLPTR );
        ASSERT_TRUE( ptr_obj1->type_id == obj1_type_id );

        /* --------------------------------------------------------------------- */

        my_obj_t* ptr_obj2 = buffer.add< my_obj_t >( obj2_type_id,
            num_d_values, num_e_values, obj2_a, obj2_b, obj2_c, obj2_d, obj2_e );

        ASSERT_TRUE( ptr_obj2 != SIXTRL_NULLPTR );
        ASSERT_TRUE( ptr_obj2->type_id == obj2_type_id );

        /* --------------------------------------------------------------------- */

        my_obj_t* ptr_obj3 = buffer.createNew< my_obj_t >(
            obj3_type_id, num_d_values, num_e_values );

        ASSERT_TRUE( ptr_obj3 != SIXTRL_NULLPTR );
        ASSERT_TRUE( ptr_obj3->type_id == obj3_type_id );

        /* --------------------------------------------------------------------- */

        st_buffer_size_t const cnt = std::fwrite(
            buffer.dataBegin< raw_t const* >(), buffer.size(), size_t{ 1 }, fp );

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

    std::vector< raw_t > base_buffer( buffer_size, raw_t{ 0 } );
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

    st::Buffer buffer( data_buffer_begin, buffer_size );

    ASSERT_TRUE( buffer.size() > size_t{ 0 } );
    ASSERT_TRUE( buffer.capacity() >= buffer_size );
    ASSERT_TRUE( buffer.getNumObjects() == size_t{ 3 } );

    obj_t const* obj_it  = buffer.indexBegin< obj_t const* >();
    obj_t const* obj_end = buffer.indexEnd< obj_t const* >();

    ASSERT_TRUE( obj_it  != nullptr );
    ASSERT_TRUE( obj_end != nullptr );
    ASSERT_TRUE( std::distance( obj_it, obj_end ) == std::ptrdiff_t{ 3 } );

    st::MyObj cmp_my_obj[ 3 ];

    st::MyObj& cmp_obj1 = cmp_my_obj[ 0 ];
    cmp_obj1.preset( obj1_type_id );
    cmp_obj1.a = obj1_a;
    cmp_obj1.b = obj1_b;
    cmp_obj1.d = &obj1_d[ 0 ];
    cmp_obj1.e = &obj1_e[ 0 ];
    std::copy( &obj1_c[ 0 ], &obj1_c[ 4 ], &cmp_obj1.c[ 0 ] );

    st::MyObj& cmp_obj2 = cmp_my_obj[ 1 ];
    cmp_obj2.preset( obj2_type_id );
    cmp_obj2.a = obj2_a;
    cmp_obj2.b = obj2_b;
    cmp_obj2.d = &obj2_d[ 0 ];
    cmp_obj2.e = &obj2_e[ 0 ];
    std::copy( &obj2_c[ 0 ], &obj2_c[ 4 ], &cmp_obj2.c[ 0 ] );

    st::MyObj& cmp_obj3 = cmp_my_obj[ 2 ];
    cmp_obj3.preset( obj3_type_id );

    st::MyObj const* cmp_obj_it = &cmp_my_obj[ 0 ];

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_obj_it )
    {
        ASSERT_TRUE( st_Object_get_type_id( obj_it ) == cmp_obj_it->type_id );
        ASSERT_TRUE( st_Object_get_size( obj_it ) > sizeof( st::MyObj ) );
        ASSERT_TRUE( st_Object_get_const_begin_ptr( obj_it ) != nullptr );

        st::MyObj const* ptr_my_obj = reinterpret_cast< st::MyObj const* >(
                st_Object_get_const_begin_ptr( obj_it ) );

        ASSERT_TRUE( ptr_my_obj != nullptr );
        ASSERT_TRUE( ptr_my_obj->type_id == st_Object_get_type_id( obj_it ) );

        ASSERT_TRUE( ptr_my_obj->a == cmp_obj_it->a );
        ASSERT_TRUE( std::fabs( ( ptr_my_obj->b - cmp_obj_it->b ) <
                        std::numeric_limits< double >::epsilon() ) );

        for( std::size_t ii = 0u ; ii < num_c_values ; ++ii )
        {
            ASSERT_TRUE( std::fabs( ( ptr_my_obj->c[ ii ] -
                cmp_obj_it->c[ ii ] ) < std::numeric_limits<
                    double >::epsilon() ) );
        }

        ASSERT_TRUE( ptr_my_obj->d != nullptr );

        if( cmp_obj_it->d != nullptr )
        {
            for( size_t ii = size_t{ 0u } ; ii < num_d_values ; ++ii )
            {
                ASSERT_TRUE( ptr_my_obj->d[ ii ] == cmp_obj_it->d[ ii ] );
            }
        }
        else
        {
            for( size_t ii = size_t{ 0u } ; ii < num_d_values ; ++ii )
            {
                ASSERT_TRUE( ptr_my_obj->d[ ii ] == uint8_t{ 0 } );
            }
        }

        ASSERT_TRUE( ptr_my_obj->e != nullptr );

        if( cmp_obj_it->e != nullptr )
        {
            for( size_t ii = size_t{ 0u } ; ii < num_e_values ; ++ii )
            {
                ASSERT_TRUE( std::fabs(
                    ptr_my_obj->e[ ii ] - cmp_obj_it->e[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
            }
        }
        else
        {
            for( size_t ii = size_t{ 0u } ; ii < num_e_values ; ++ii )
            {
                ASSERT_TRUE( std::fabs( ptr_my_obj->e[ ii ] - double{ 0.0 } ) <
                        std::numeric_limits< double >::epsilon() );
            }
        }
    }
}

TEST( CXX_CommonBufferTests, NewBufferAndGrowingWithinCapacity )
{
    namespace st = sixtrack;

    using type_id_t = ::st_object_type_id_t;
    using buffer_t  = st::Buffer;
    using size_t    = buffer_t::size_type;

    constexpr size_t NUM_OBJECTS          = size_t{ 3u };
    constexpr size_t NUM_SLOTS            = size_t{ 48u };
    constexpr size_t NUM_DATAPTRS_PER_OBJ = size_t{ 2u };
    constexpr size_t NUM_DATAPTRS         = NUM_OBJECTS * NUM_DATAPTRS_PER_OBJ;
    constexpr size_t NUM_GARBAGE_RANGES   = size_t{ 1u };

//     constexpr size_t num_c_values         = size_t{ 4 };
    constexpr size_t num_d_values         = size_t{ 4 };
    constexpr size_t num_e_values         = size_t{ 2 };

    size_t const buffer_capacity = st::Buffer::CalculateBufferSize(
        NUM_OBJECTS, NUM_SLOTS, NUM_DATAPTRS, NUM_GARBAGE_RANGES );

    st::Buffer buffer( buffer_capacity );

    ASSERT_TRUE(  buffer.hasDataStore() );
    ASSERT_TRUE(  buffer.usesDataStore() );
    ASSERT_TRUE(  buffer.ownsDataStore() );
    ASSERT_TRUE(  buffer.allowModifyContents() );
    ASSERT_TRUE(  buffer.allowClear() );
    ASSERT_TRUE(  buffer.allowAppend() );
    ASSERT_TRUE(  buffer.allowResize() );

    ASSERT_TRUE(  buffer.usesMempoolDataStore() );
    ASSERT_TRUE( !buffer.usesSpecialOpenCLDataStore() );
    ASSERT_TRUE( !buffer.usesSpecialCudaDataStore() );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE(  buffer.getNumObjects()  == size_t{ 0 } );
    ASSERT_TRUE(  buffer.getNumSlots()    == size_t{ 0 } );
    ASSERT_TRUE(  buffer.getNumDataptrs() == size_t{ 0 } );

    type_id_t const obj1_type_id   = type_id_t{ 3 };
    int32_t const   obj1_a         = int32_t{ 25 };
    double  const   obj1_b         = double{ 26.0 };
    double  const   obj1_c[]       = { 26.0, 27.0, 28.0, 29.0 };
    uint8_t obj1_d[ num_d_values ] = { 31, 32, 33, 34 };
    double  obj1_e[ num_e_values ] = { 35.0, 36.0 };

    st::MyObj* ptr_obj1 = buffer.add< st::MyObj >(
        obj1_type_id, num_d_values, num_e_values,
            obj1_a, obj1_b, obj1_c, obj1_d, obj1_e );

    ASSERT_TRUE( ptr_obj1 != nullptr );
    ASSERT_TRUE( ptr_obj1->type_id == obj1_type_id );
    ASSERT_TRUE( ptr_obj1->a       == obj1_a );

    ASSERT_TRUE( buffer.getNumObjects()  == size_t{ 1 } );
    ASSERT_TRUE( buffer.getNumDataptrs() == NUM_DATAPTRS_PER_OBJ );
    ASSERT_TRUE( buffer.getNumSlots()    >  size_t{ 0 } );

    size_t const num_slots_after_obj1 = buffer.getNumSlots();

    /* --------------------------------------------------------------------- */

    type_id_t const obj2_type_id   = type_id_t{ 4 };
    int32_t const   obj2_a         = int32_t{ 100 };
    double  const   obj2_b         = double{ 101.0 };
    double  const   obj2_c[]       = { 102.0, 103.0, 104.0, 105.0 };
    uint8_t obj2_d[ num_d_values ] = { 106, 107, 108, 109 };
    double  obj2_e[ num_e_values ] = { 110.0, 111.0 };

    st::MyObj* ptr_obj2 = buffer.add< st::MyObj >(
        obj2_type_id, num_d_values, num_e_values,
            obj2_a, obj2_b, obj2_c, obj2_d, obj2_e );

    ASSERT_TRUE( ptr_obj2 != nullptr );
    ASSERT_TRUE( ptr_obj2->type_id == obj2_type_id );
    ASSERT_TRUE( ptr_obj2->a       == obj2_a );

    ASSERT_TRUE( buffer.getNumObjects()  == size_t{ 2 } );
    ASSERT_TRUE( buffer.getNumDataptrs() == size_t{ 2 } * NUM_DATAPTRS_PER_OBJ );
    ASSERT_TRUE( buffer.getNumSlots()    >  num_slots_after_obj1 );

    size_t const num_slots_after_obj2 = buffer.getNumSlots();

    /* --------------------------------------------------------------------- */

    type_id_t const obj3_type_id = type_id_t{ 5 };

    st::MyObj* ptr_obj3 = buffer.createNew< st::MyObj >(
        obj3_type_id, num_d_values, num_e_values );

    ASSERT_TRUE( ptr_obj3 != nullptr );
    ASSERT_TRUE( ptr_obj3->type_id == obj3_type_id );

    ASSERT_TRUE( buffer.getNumObjects()  == size_t{ 3 } );
    ASSERT_TRUE( buffer.getNumDataptrs() == size_t{ 3 } * NUM_DATAPTRS_PER_OBJ );
    ASSERT_TRUE( buffer.getNumSlots()    >  num_slots_after_obj2 );
}

TEST( CXX_CommonBufferTests, AddGenericObjectsTestAutoGrowingOfBuffer )
{
    namespace st = sixtrack;

    using buf_size_t    = st::Buffer::size_type;
    using type_id_t     = st::Buffer::type_id_t;
    using generic_obj_t = ::st_GenericObj;
    using index_obj_t   = ::st_Object;

    st::Buffer buffer;

    ASSERT_TRUE( buffer.allowResize() );
    ASSERT_TRUE( buffer.allowAppend() );
    ASSERT_TRUE( buffer.allowRemap()  );

    buf_size_t prev_capacity     = buffer.capacity();
    buf_size_t prev_size         = buffer.size();
    buf_size_t prev_num_objects  = buffer.getNumObjects();
    buf_size_t prev_num_slots    = buffer.getNumSlots();
    buf_size_t prev_num_dataptrs = buffer.getNumDataptrs();

    ASSERT_TRUE( prev_size         >  buf_size_t{ 0 } );
    ASSERT_TRUE( prev_capacity     >= prev_size );
    ASSERT_TRUE( prev_num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( prev_num_dataptrs == buf_size_t{ 0 } );

    constexpr buf_size_t NUM_OBJECTS_TO_ADD = buf_size_t{ 100 };
    constexpr buf_size_t NUM_DATAPTRS       = buf_size_t{   2 };
    constexpr buf_size_t num_d_values       = buf_size_t{  10 };
    constexpr buf_size_t num_e_values       = buf_size_t{  10 };

    buf_size_t offsets[ NUM_DATAPTRS ] =
    {
        offsetof( generic_obj_t, d ), offsetof( generic_obj_t, e )
    };

    buf_size_t sizes[ NUM_DATAPTRS ] =
    {
        sizeof( uint8_t ), sizeof( double )
    };

    buf_size_t counts[ NUM_DATAPTRS ] =
    {
        num_d_values, num_e_values
    };

    generic_obj_t temp;
    temp.a       = int32_t{ 0 };
    temp.b       = double{ 0.0 };
    temp.num_d   = num_d_values;
    temp.d       = nullptr;
    temp.num_e   = num_e_values;
    temp.e       = nullptr;

    std::fill( &temp.c[ 0 ], &temp.c[ 4 ], double{ 0.0 } );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS_TO_ADD ; ++ii )
    {
        temp.type_id = static_cast< type_id_t >( ii );

        index_obj_t* index_obj = buffer.addObject( temp, temp.type_id,
                NUM_DATAPTRS, offsets, sizes, counts );

        ASSERT_TRUE( index_obj != nullptr );
        ASSERT_TRUE( ::st_Object_get_type_id( index_obj )   == temp.type_id );
        ASSERT_TRUE( ::st_Object_get_begin_ptr( index_obj ) != nullptr );

        buf_size_t const capacity = buffer.capacity();
        buf_size_t const size     = buffer.size();

        ASSERT_TRUE( capacity >= prev_capacity );
        ASSERT_TRUE( size     >  prev_size     );
        ASSERT_TRUE( capacity >= size  );

        prev_capacity = capacity;
        prev_size     = size;

        buf_size_t const num_objects  = buffer.getNumObjects();
        buf_size_t const num_slots    = buffer.getNumSlots();
        buf_size_t const num_dataptrs = buffer.getNumDataptrs();

        ASSERT_TRUE( num_objects  == prev_num_objects + buf_size_t{ 1 } );
        ASSERT_TRUE( num_slots    >= prev_num_slots );
        ASSERT_TRUE( num_dataptrs >= prev_num_dataptrs );

        prev_num_objects  = num_objects;
        prev_num_slots    = num_slots;
        prev_num_dataptrs = num_dataptrs;
    }

    ASSERT_TRUE( buffer.getNumObjects() == NUM_OBJECTS_TO_ADD );
}

TEST( CXX_CommonBufferTests, DumpToFileConstructFromDumpCompare)
{
    namespace st = sixtrack;

    using buf_size_t  = st::Buffer::size_type;
    using prng_seed_t = unsigned long long;

    prng_seed_t const seed = prng_seed_t{ 20181105 };
    ::st_Random_init_genrand64( seed );

    st::Buffer buffer;

    buf_size_t const NUM_OBJECTS  = buf_size_t{ 100 };
    buf_size_t const num_d_values = buf_size_t{ 10  };
    buf_size_t const num_e_values = buf_size_t{ 10  };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        ::st_object_type_id_t const type_id =
            static_cast< ::st_object_type_id_t >( ii );

        ::st_GenericObj* obj = ::st_GenericObj_new(
                &buffer, type_id, num_d_values, num_e_values );

        ASSERT_TRUE( obj != nullptr );
        ::st_GenericObj_init_random( obj );
    }

    ASSERT_TRUE( buffer.size()           >  buf_size_t{ 0 } );
    ASSERT_TRUE( buffer.capacity()       >= buffer.size() );
    ASSERT_TRUE( buffer.getNumObjects()  == NUM_OBJECTS );

    std::string const temp_dump_file( "./temp_binary_dump.bin" );
    double const ABS_TRESHOLD = std::numeric_limits< double >::epsilon();

    ASSERT_TRUE( buffer.writeToFile( temp_dump_file ) );

    st::Buffer cmp_buffer( temp_dump_file );
    ASSERT_TRUE( cmp_buffer.getNumObjects() == NUM_OBJECTS );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        auto orig_ptr     = buffer[ ii ];
        auto restored_ptr = cmp_buffer[ ii ];

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( restored_ptr != nullptr );
        ASSERT_TRUE( restored_ptr != orig_ptr );

        ::st_GenericObj const* orig_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( orig_ptr ) );

        ::st_GenericObj const* restored_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr(
                restored_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( restored_obj != nullptr );
        ASSERT_TRUE( restored_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::st_GenericObj_compare_values( orig_obj, restored_obj ) ) ||
            ( 0 == ::st_GenericObj_compare_values_with_treshold(
                orig_obj, restored_obj, ABS_TRESHOLD ) ) );
    }

    std::remove( temp_dump_file.c_str() );
}


TEST( CXX_CommonBufferTests, WriteBufferNormalizedAddrRestoreVerify )
{
    namespace st = sixtrack;

    using buf_size_t  = st::Buffer::size_type;
    using address_t   = st::Buffer::address_t;
    using prng_seed_t = unsigned long long;

    prng_seed_t const seed = prng_seed_t{ 20181105 };
    ::st_Random_init_genrand64( seed );

    st::Buffer cmp_buffer;
    st::Buffer temp_buffer;

    buf_size_t const NUM_OBJECTS  = buf_size_t{ 100 };
    buf_size_t const num_d_values = buf_size_t{ 10  };
    buf_size_t const num_e_values = buf_size_t{ 10  };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        ::st_object_type_id_t const type_id =
            static_cast< ::st_object_type_id_t >( ii );

        ::st_GenericObj* obj = ::st_GenericObj_new(
                &cmp_buffer, type_id, num_d_values, num_e_values );

        ASSERT_TRUE( obj != nullptr );
        ::st_GenericObj_init_random( obj );

        ::st_GenericObj* copy_obj = ::st_GenericObj_add_copy(
                &temp_buffer, obj );

        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != obj );
    }

    double const ABS_TRESHOLD = std::numeric_limits< double >::epsilon();

    ASSERT_TRUE( cmp_buffer.getNumObjects()  == NUM_OBJECTS );
    ASSERT_TRUE( temp_buffer.getNumObjects() == NUM_OBJECTS );
    ASSERT_TRUE( temp_buffer.size() == cmp_buffer.size() );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        auto orig_ptr = cmp_buffer[ ii ];
        auto copy_ptr = temp_buffer[ ii ];

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != orig_ptr );

        ::st_GenericObj const* orig_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( orig_ptr ) );

        ::st_GenericObj const* copy_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( copy_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::st_GenericObj_compare_values( orig_obj, copy_obj ) ) ||
            ( 0 == ::st_GenericObj_compare_values_with_treshold(
                orig_obj, copy_obj, ABS_TRESHOLD ) ) );
    }

    /* Write file and remap the contents to a normalized base address: */

    address_t const base_addr = temp_buffer.getDataBeginAddr();

    address_t const target_addr = ( base_addr != address_t { 0x1000 } )
        ? address_t { 0x1000 } : address_t { 0x2000 };

    std::string const path_to_temp_file( "./temp_norm_addr_cxx.bin" );

    ASSERT_TRUE( temp_buffer.writeToFileNormalizedAddr(
        path_to_temp_file, target_addr ) );

    /* repeat the previous checks to verify that the normaliezd write to a
     * file has not changed temp_buffer */

    ASSERT_TRUE( base_addr == temp_buffer.getDataBeginAddr() );
    ASSERT_TRUE( temp_buffer.getNumObjects() == NUM_OBJECTS );
    ASSERT_TRUE( cmp_buffer.size() == temp_buffer.size() );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        auto orig_ptr = cmp_buffer[ ii ];
        auto copy_ptr = temp_buffer[ ii ];

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != nullptr );
        ASSERT_TRUE( copy_ptr != orig_ptr );

        ::st_GenericObj const* orig_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( orig_ptr ) );

        ::st_GenericObj const* copy_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( copy_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( copy_obj != nullptr );
        ASSERT_TRUE( copy_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::st_GenericObj_compare_values( orig_obj, copy_obj ) ) ||
            ( 0 == ::st_GenericObj_compare_values_with_treshold(
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

    st::Buffer restored_buffer( path_to_temp_file );
    std::remove( path_to_temp_file.c_str() );

    ASSERT_TRUE( !restored_buffer.needsRemapping() );
    ASSERT_TRUE(  restored_buffer.getNumObjects() == NUM_OBJECTS );
    ASSERT_TRUE(  restored_buffer.size() == cmp_buffer.size() );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_OBJECTS ; ++ii )
    {
        auto orig_ptr = cmp_buffer[ ii ];
        auto rest_ptr = restored_buffer[ ii ];

        ASSERT_TRUE( orig_ptr != nullptr );
        ASSERT_TRUE( rest_ptr != nullptr );
        ASSERT_TRUE( rest_ptr != orig_ptr );

        ::st_GenericObj const* orig_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( orig_ptr ) );

        ::st_GenericObj const* rest_obj = reinterpret_cast<
            ::st_GenericObj const* >( ::st_Object_get_const_begin_ptr( rest_ptr ) );

        ASSERT_TRUE( orig_obj != nullptr );
        ASSERT_TRUE( rest_obj != nullptr );
        ASSERT_TRUE( rest_obj != orig_obj );

        ASSERT_TRUE(
            ( 0 == ::st_GenericObj_compare_values( orig_obj, rest_obj ) ) ||
            ( 0 == ::st_GenericObj_compare_values_with_treshold(
                orig_obj, rest_obj, ABS_TRESHOLD ) ) );
    }
}

/* ************************************************************************* */

/* end: tests/sixtracklib/common/test_buffer_cxx.cpp */
