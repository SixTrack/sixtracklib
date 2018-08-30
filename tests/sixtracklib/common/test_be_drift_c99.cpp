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

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"

#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/impl/be_drift.h"

/* ************************************************************************* *
 * ******  st_Drift:
 * ************************************************************************* */

TEST( C99_CommonBeamElementDriftTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t   = ::st_buffer_size_t;
    using object_t = ::NS(Object);
    using raw_t    = unsigned char;
    using elem_t   = ::st_Drift;

    ::st_Buffer* eb = ::st_Buffer_new( size_t{ 1u << 20u } );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    elem_t* ptr_drift1 = ::st_Drift_new( eb );

    ASSERT_TRUE( ptr_drift1 != nullptr );
    ASSERT_TRUE( std::fabs( ::st_Drift_get_length( ptr_drift1 ) - ZERO ) < EPS );

    double len = double{ 1.0 };
    ::st_Drift_set_length( ptr_drift1, len );
    ASSERT_TRUE( std::fabs( ::st_Drift_get_length( ptr_drift1 )  - len ) < EPS );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == size_t{ 1 } );

    len = double{ 2.0 };
    elem_t* ptr_drift2 = ::st_Drift_add( eb, len );
    ASSERT_TRUE( ptr_drift2 != nullptr );
    ASSERT_TRUE( std::fabs( ::st_Drift_get_length( ptr_drift2 ) - len ) < EPS );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == size_t{ 2 } );

    constexpr size_t const NUM_DRIFTS = size_t{ 100u };

    len += double{ 1.0 };
    size_t ii = ::st_Buffer_get_num_of_objects( eb );

    for( ; ii < NUM_DRIFTS ; ++ii, len += double{ 1.0 } )
    {
        size_t const jj = ii + size_t{ 1 };
        elem_t* drift = ::st_Drift_add( eb, len );

        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::fabs( ::st_Drift_get_length( drift ) - len ) < EPS );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == jj );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ::st_Buffer_get_size( eb ) > size_t{ 0 } );

    std::vector< raw_t > data_buffer( ::st_Buffer_get_size( eb ), raw_t{ 0 } );
    data_buffer.assign( ::st_Buffer_get_const_data_begin( eb ),
                        ::st_Buffer_get_const_data_end( eb ) );

    ::st_Buffer cmp_buffer;
    ::st_Buffer_preset( &cmp_buffer );
    int success = ::st_Buffer_init(
        &cmp_buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) ==
                 ::st_Buffer_get_num_of_objects( &cmp_buffer ) );

    object_t const* obj_it  = ::st_Buffer_get_const_objects_begin( eb );
    object_t const* obj_end = ::st_Buffer_get_const_objects_end( eb );
    object_t const* cmp_it  = ::st_Buffer_get_const_objects_begin( &cmp_buffer );

    len = double{ 1.0 };

    SIXTRL_CONSTEXPR_OR_CONST ::st_object_type_id_t
        BEAM_ELEMENT_TYPE_ID = NS(OBJECT_TYPE_DRIFT);

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

        ASSERT_TRUE( std::fabs( ::st_Drift_get_length( elem ) - len ) < EPS );
        ASSERT_TRUE( std::fabs( ::st_Drift_get_length( cmp_elem ) - len ) < EPS );
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_free( &cmp_buffer );
}

/* ************************************************************************* *
 * ******  st_DriftExact:
 * ************************************************************************* */

TEST( C99_CommonBeamElementDriftExactTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t    = ::st_buffer_size_t;
    using object_t  = ::NS(Object);
    using raw_t     = unsigned char;
    using elem_t    = ::st_DriftExact;
    using type_id_t = ::st_object_type_id_t;

    ::st_Buffer* eb = ::st_Buffer_new( size_t{ 1u << 20u } );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    elem_t* ptr_drift1 = ::st_DriftExact_new( eb );

    ASSERT_TRUE( ptr_drift1 != nullptr );
    ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( ptr_drift1 ) - ZERO ) < EPS );

    double len = double{ 1.0 };
    ::st_DriftExact_set_length( ptr_drift1, len );
    ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( ptr_drift1 )  - len ) < EPS );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == size_t{ 1 } );

    len = double{ 2.0 };
    elem_t* ptr_drift2 = ::st_DriftExact_add( eb, len );
    ASSERT_TRUE( ptr_drift2 != nullptr );
    ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( ptr_drift2 ) - len ) < EPS );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == size_t{ 2 } );

    constexpr size_t const NUM_DRIFTS = size_t{ 100u };

    len += double{ 1.0 };
    size_t ii = ::st_Buffer_get_num_of_objects( eb );

    for( ; ii < NUM_DRIFTS ; ++ii, len += double{ 1.0 } )
    {
        size_t const jj = ii + size_t{ 1 };
        elem_t* drift = ::st_DriftExact_add( eb, len );

        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( drift ) - len ) < EPS );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == jj );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( ::st_Buffer_get_size( eb ) > size_t{ 0 } );

    std::vector< raw_t > data_buffer( ::st_Buffer_get_size( eb ), raw_t{ 0 } );
    data_buffer.assign( ::st_Buffer_get_const_data_begin( eb ),
                        ::st_Buffer_get_const_data_end( eb ) );

    ::st_Buffer cmp_buffer;
    ::st_Buffer_preset( &cmp_buffer );
    int success = ::st_Buffer_init(
        &cmp_buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) ==
                 ::st_Buffer_get_num_of_objects( &cmp_buffer ) );

    object_t const* obj_it  = ::st_Buffer_get_const_objects_begin( eb );
    object_t const* obj_end = ::st_Buffer_get_const_objects_end( eb );
    object_t const* cmp_it  = ::st_Buffer_get_const_objects_begin( &cmp_buffer );

    len = double{ 1.0 };

    SIXTRL_CONSTEXPR_OR_CONST type_id_t
        BEAM_ELEMENT_TYPE_ID = NS(OBJECT_TYPE_DRIFT_EXACT );

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

        ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( elem ) - len ) < EPS );
        ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( cmp_elem ) - len ) < EPS );
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_free( &cmp_buffer );
}

/* end: tests/sixtracklib/common/test_be_drift_c99.cpp  */
