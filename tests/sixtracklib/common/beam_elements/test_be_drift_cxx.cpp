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
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/be_drift/be_drift.hpp"

/* ************************************************************************* *
 * ******  sixtrack::Drift:
 * ************************************************************************* */

TEST( CXXCommonBeamElementDriftTests, MinimalAddToBufferCopyRemapRead )
{
    namespace st    = sixtrack;

    using buffer_t  = st::Buffer;
    using size_t    = buffer_t::size_type;
    using object_t  = buffer_t::object_t;
    using type_id_t = buffer_t::type_id_t;
    using raw_t     = unsigned char;
    using elem_t    = st::Drift;

    st::Buffer eb( size_t{ 1u << 20u } );
    ASSERT_TRUE( eb.size() > size_t{ 0 } );
    ASSERT_TRUE( eb.capacity() > size_t{ 0 } );
    ASSERT_TRUE( eb.hasDataStore()  );
    ASSERT_TRUE( eb.usesDataStore() );
    ASSERT_TRUE( eb.ownsDataStore() );

    /* --------------------------------------------------------------------- */

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    double len = ZERO;
    size_t num_objects = size_t{ 0 };

    elem_t dummy;
    dummy.setLength( len );

    size_t const* offsets = nullptr;
    size_t const* sizes   = nullptr;
    size_t const* counts  = nullptr;

    object_t* ptr_object = eb.addObject(
        dummy, ::st_OBJECT_TYPE_DRIFT, size_t{ 0 }, offsets, sizes, counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::st_Object_get_const_begin_ptr( ptr_object ) != nullptr );
    ASSERT_TRUE( ::st_Object_get_size( ptr_object ) >= sizeof( dummy ) );
    ASSERT_TRUE( ::st_Object_get_type_id( ptr_object ) ==
                 ::st_OBJECT_TYPE_DRIFT );

    ++num_objects;
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );

    elem_t* ptr_drift0 = reinterpret_cast< elem_t* >(
        ::st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_drift0 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift0->getLength() - len ) < EPS );



    len += double{ 1.0 };
    elem_t* ptr_drift1 = st::Drift_new( eb );

    ASSERT_TRUE( ptr_drift1 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift1->getLength() - ZERO ) < EPS );

    ptr_drift1->setLength( len );
    ASSERT_TRUE( std::fabs( ptr_drift1->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );



    len += double{ 1.0 };
    elem_t* ptr_drift2 = st::Drift_add( eb, len );

    ASSERT_TRUE( ptr_drift2 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift2->getLength() - len ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );



    len += double{ 1.0 };
    elem_t* ptr_drift3 = eb.createNew< st::Drift >();
    ASSERT_TRUE( ptr_drift3 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift3->getLength() - ZERO ) < EPS );

    ptr_drift3->setLength( len );
    ASSERT_TRUE( std::fabs( ptr_drift3->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );


    len += double{ 1.0 };
    elem_t* ptr_drift4 = eb.add< st::Drift >( len );
    ASSERT_TRUE( ptr_drift4 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift4->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );

    constexpr size_t const NUM_DRIFTS = size_t{ 100u };

    len += double{ 1.0 };
    size_t ii = eb.getNumObjects();

    for( ; ii < NUM_DRIFTS ; ++ii, len += double{ 1.0 } )
    {
        elem_t* drift = st::Drift_add( eb, len );

        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::fabs( drift->getLength() - len ) < EPS );
        ASSERT_TRUE( eb.getNumObjects() == num_objects++ );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( eb.size() > size_t{ 0 } );

    std::vector< raw_t > data_buffer( eb.size(), raw_t{ 0 } );
    data_buffer.assign( eb.dataBegin< raw_t const* >(),
                        eb.dataEnd< raw_t const* >() );

    st::Buffer cmp_buffer( data_buffer.data(), data_buffer.size() );
    ASSERT_TRUE( eb.getNumObjects() == cmp_buffer.getNumObjects() );

    object_t const* obj_it  = eb.indexBegin< object_t const* >();
    object_t const* obj_end = eb.indexEnd<   object_t const* >();
    object_t const* cmp_it  = cmp_buffer.indexBegin< object_t const* >();

    len = ZERO;

    constexpr type_id_t BEAM_ELEMENT_TYPE_ID = ::st_OBJECT_TYPE_DRIFT;

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it, len += double{ 1.0 } )
    {
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) == BEAM_ELEMENT_TYPE_ID );
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) ==
                     ::st_Object_get_type_id( cmp_it ) );

        ASSERT_TRUE( ::st_Object_get_size( obj_it ) >= sizeof( elem_t ) );
        ASSERT_TRUE( ::st_Object_get_size( obj_it ) ==
                     ::st_Object_get_size( cmp_it ) );

        elem_t const* elem = reinterpret_cast< elem_t const* >(
            ::st_Object_get_const_begin_ptr( obj_it ) );

        elem_t const* cmp_elem = reinterpret_cast< elem_t const* >(
            ::st_Object_get_const_begin_ptr( cmp_it ) );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( elem->getLength()     - len ) < EPS );
        ASSERT_TRUE( std::fabs( cmp_elem->getLength() - len ) < EPS );
    }
}

/* ************************************************************************* *
 * ******  st_DriftExact:
 * ************************************************************************* */

TEST( CXXCommonBeamElementDriftExactTests, MinimalAddToBufferCopyRemapRead )
{
    namespace st    = sixtrack;

    using buffer_t  = st::Buffer;
    using size_t    = buffer_t::size_type;
    using object_t  = buffer_t::object_t;
    using type_id_t = buffer_t::type_id_t;
    using raw_t     = unsigned char;
    using elem_t    = st::DriftExact;

    st::Buffer eb( size_t{ 1u << 20u } );
    ASSERT_TRUE( eb.size() > size_t{ 0 } );
    ASSERT_TRUE( eb.capacity() > size_t{ 0 } );
    ASSERT_TRUE( eb.hasDataStore()  );
    ASSERT_TRUE( eb.usesDataStore() );
    ASSERT_TRUE( eb.ownsDataStore() );

    /* --------------------------------------------------------------------- */

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    double len = ZERO;
    size_t num_objects = size_t{ 0 };

    elem_t dummy;
    dummy.setLength( len );

    size_t const* offsets = nullptr;
    size_t const* sizes   = nullptr;
    size_t const* counts  = nullptr;

    object_t* ptr_object = eb.addObject( dummy, ::st_OBJECT_TYPE_DRIFT_EXACT,
            size_t{ 0 }, offsets, sizes, counts );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::st_Object_get_const_begin_ptr( ptr_object ) != nullptr );
    ASSERT_TRUE( ::st_Object_get_size( ptr_object ) >= sizeof( dummy ) );
    ASSERT_TRUE( ::st_Object_get_type_id( ptr_object ) ==
                 ::st_OBJECT_TYPE_DRIFT_EXACT );

    ++num_objects;
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );

    elem_t* ptr_drift0 = reinterpret_cast< elem_t* >(
        ::st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_drift0 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift0->getLength() - len ) < EPS );



    len += double{ 1.0 };
    elem_t* ptr_drift1 = st::DriftExact_new( eb );

    ASSERT_TRUE( ptr_drift1 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift1->getLength() - ZERO ) < EPS );

    ptr_drift1->setLength( len );
    ASSERT_TRUE( std::fabs( ptr_drift1->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );



    len += double{ 1.0 };
    elem_t* ptr_drift2 = st::DriftExact_add( eb, len );

    ASSERT_TRUE( ptr_drift2 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift2->getLength() - len ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );



    len += double{ 1.0 };
    elem_t* ptr_drift3 = eb.createNew< st::DriftExact >();
    ASSERT_TRUE( ptr_drift3 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift3->getLength() - ZERO ) < EPS );

    ptr_drift3->setLength( len );
    ASSERT_TRUE( std::fabs( ptr_drift3->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );


    len += double{ 1.0 };
    elem_t* ptr_drift4 = eb.add< st::DriftExact >( len );
    ASSERT_TRUE( ptr_drift4 != nullptr );
    ASSERT_TRUE( std::fabs( ptr_drift4->getLength() - len  ) < EPS );
    ASSERT_TRUE( eb.getNumObjects() == num_objects++ );

    constexpr size_t const NUM_DRIFTS = size_t{ 100u };

    len += double{ 1.0 };
    size_t ii = eb.getNumObjects();

    for( ; ii < NUM_DRIFTS ; ++ii, len += double{ 1.0 } )
    {
        elem_t* drift = st::DriftExact_add( eb, len );

        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::fabs( drift->getLength() - len ) < EPS );
        ASSERT_TRUE( eb.getNumObjects() == num_objects++ );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( eb.size() > size_t{ 0 } );

    std::vector< raw_t > data_buffer( eb.size(), raw_t{ 0 } );
    data_buffer.assign( eb.dataBegin< raw_t const* >(),
                        eb.dataEnd< raw_t const* >() );

    st::Buffer cmp_buffer( data_buffer.data(), data_buffer.size() );
    ASSERT_TRUE( eb.getNumObjects() == cmp_buffer.getNumObjects() );

    object_t const* obj_it  = eb.indexBegin< object_t const* >();
    object_t const* obj_end = eb.indexEnd<   object_t const* >();
    object_t const* cmp_it  = cmp_buffer.indexBegin< object_t const* >();

    len = ZERO;

    constexpr type_id_t BEAM_ELEMENT_TYPE_ID = ::st_OBJECT_TYPE_DRIFT_EXACT;

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it, len += double{ 1.0 } )
    {
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) == BEAM_ELEMENT_TYPE_ID );
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) ==
                     ::st_Object_get_type_id( cmp_it ) );

        ASSERT_TRUE( ::st_Object_get_size( obj_it ) >= sizeof( elem_t ) );
        ASSERT_TRUE( ::st_Object_get_size( obj_it ) ==
                     ::st_Object_get_size( cmp_it ) );

        elem_t const* elem = reinterpret_cast< elem_t const* >(
            ::st_Object_get_const_begin_ptr( obj_it ) );

        elem_t const* cmp_elem = reinterpret_cast< elem_t const* >(
            ::st_Object_get_const_begin_ptr( cmp_it ) );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( elem->getLength()     - len ) < EPS );
        ASSERT_TRUE( std::fabs( cmp_elem->getLength() - len ) < EPS );
    }
}

/* end: tests/sixtracklib/common/test_be_drift_cxx.cpp  */
