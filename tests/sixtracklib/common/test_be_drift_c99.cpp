#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
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
    using object_t = ::st_Object;
    using raw_t    = unsigned char;
    using belem_t  = ::st_Drift;
    using real_t   = SIXTRL_REAL_T;

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    /* --------------------------------------------------------------------- */

    std::mt19937_64::result_type const seed = 20180830u;

    std::mt19937_64 prng;
    prng.seed( seed );

    using len_dist_t = std::uniform_real_distribution< real_t >;

    len_dist_t length_dist( real_t{ 0.0 }, real_t{ +10.0 } );

    static SIXTRL_CONSTEXPR_OR_CONST size_t
        NUM_BEAM_ELEMENTS = size_t{ 1000 };

    ::st_object_type_id_t const BEAM_ELEMENT_TYPE_ID = ::st_OBJECT_TYPE_DRIFT;
    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    size_t const slot_size      = ::st_BUFFER_DEFAULT_SLOT_SIZE;
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };
    size_t const num_dataptrs   = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        real_t const length = length_dist( prng );

        belem_t* ptr_drift = ::st_Drift_preset( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_drift != nullptr );
        ::st_Drift_set_length( ptr_drift, length );

        ASSERT_TRUE( std::fabs( length - ::st_Drift_get_length( ptr_drift ) ) < EPS );

        num_slots += ::st_ManagedBuffer_predict_required_num_slots( nullptr,
            sizeof( ::st_Drift ), ::st_Drift_get_num_dataptrs( ptr_drift ),
                nullptr, nullptr, slot_size );
    }

    /* --------------------------------------------------------------------- */

    size_t const requ_buffer_size = ::st_ManagedBuffer_calculate_buffer_length(
        nullptr, num_objs, num_slots, num_dataptrs, num_garbage, slot_size );

    ::st_Buffer* eb = ::st_Buffer_new( requ_buffer_size );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    size_t   be_index = size_t{ 0 };
    belem_t* ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    object_t* ptr_object = ::st_Buffer_add_object( eb, ptr_orig, sizeof( belem_t ),
        BEAM_ELEMENT_TYPE_ID, ::st_Drift_get_num_dataptrs( ptr_orig ),
            nullptr, nullptr, nullptr );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
    ASSERT_TRUE( ::st_Object_get_const_begin_ptr( ptr_object ) != nullptr );
    ASSERT_TRUE( ::st_Object_get_size( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( ::st_Object_get_type_id( ptr_object ) == BEAM_ELEMENT_TYPE_ID );

    belem_t* ptr_drift = reinterpret_cast< belem_t* >(
        ::st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( EPS > std::fabs( ::st_Drift_get_length( ptr_drift ) -
                                  ::st_Drift_get_length( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = ::st_Drift_new( eb );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs( ::st_Drift_get_length( ptr_drift ) - ZERO ) );

    ::st_Drift_set_length( ptr_drift, ::st_Drift_get_length( ptr_orig ) );
    ASSERT_TRUE( EPS > std::fabs( ::st_Drift_get_length( ptr_drift ) -
                                  ::st_Drift_get_length( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = ::st_Drift_add( eb, ::st_Drift_get_length( ptr_orig ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
    ASSERT_TRUE( EPS > std::fabs( ::st_Drift_get_length( ptr_drift ) -
                                  ::st_Drift_get_length( ptr_orig  ) ) );

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig  = &orig_beam_elements[ be_index++ ];
        ptr_drift = ::st_Drift_add( eb, ::st_Drift_get_length( ptr_orig ) );

        ASSERT_TRUE( ptr_drift != nullptr );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
        ASSERT_TRUE( EPS > std::fabs( ::st_Drift_get_length( ptr_drift ) -
                                      ::st_Drift_get_length( ptr_orig  ) ) );
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

    be_index = size_t{ 0 };

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
    {
        ptr_orig = &orig_beam_elements[ be_index++ ];

        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) == BEAM_ELEMENT_TYPE_ID );
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) ==
                     ::st_Object_get_type_id( cmp_it ) );

        ASSERT_TRUE( ::st_Object_get_size( obj_it ) >= sizeof( belem_t ) );
        ASSERT_TRUE( ::st_Object_get_size( obj_it ) ==
                     ::st_Object_get_size( cmp_it ) );

        belem_t const* elem = reinterpret_cast< belem_t const* >(
            ::st_Object_get_const_begin_ptr( obj_it ) );

        belem_t const* cmp_elem = reinterpret_cast< belem_t const* >(
            ::st_Object_get_const_begin_ptr( cmp_it ) );

        ASSERT_TRUE( ptr_orig != elem );
        ASSERT_TRUE( ptr_orig != cmp_elem );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( ::st_Drift_get_length( elem ) -
                                ::st_Drift_get_length( ptr_orig ) ) < EPS );

        ASSERT_TRUE( std::fabs( ::st_Drift_get_length( cmp_elem ) -
                                ::st_Drift_get_length( ptr_orig ) ) < EPS );
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
    using size_t   = ::st_buffer_size_t;
    using object_t = ::st_Object;
    using raw_t    = unsigned char;
    using belem_t  = ::st_DriftExact;
    using real_t   = SIXTRL_REAL_T;

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    /* --------------------------------------------------------------------- */

    std::mt19937_64::result_type const seed = 20180830u;

    std::mt19937_64 prng;
    prng.seed( seed );

    using len_dist_t = std::uniform_real_distribution< real_t >;

    len_dist_t length_dist( real_t{ 0.0 }, real_t{ +10.0 } );

    static SIXTRL_CONSTEXPR_OR_CONST size_t
        NUM_BEAM_ELEMENTS = size_t{ 1000 };

    ::st_object_type_id_t const BEAM_ELEMENT_TYPE_ID = ::st_OBJECT_TYPE_DRIFT_EXACT;
    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    size_t const slot_size      = ::st_BUFFER_DEFAULT_SLOT_SIZE;
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };
    size_t const num_dataptrs   = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        real_t  const length   = length_dist( prng );

        belem_t* ptr_drift = ::st_DriftExact_preset( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_drift != nullptr );
        ::st_DriftExact_set_length( ptr_drift, length );

        ASSERT_TRUE( EPS > std::fabs( length -
            ::st_DriftExact_get_length( ptr_drift ) ) );

        num_slots += ::st_ManagedBuffer_predict_required_num_slots( nullptr,
            sizeof( ::st_DriftExact ),
                ::st_DriftExact_get_num_dataptrs( ptr_drift ), nullptr,
                    nullptr, slot_size );
    }

    /* --------------------------------------------------------------------- */

    size_t const requ_buffer_size = ::st_ManagedBuffer_calculate_buffer_length(
        nullptr, num_objs, num_slots, num_dataptrs, num_garbage, slot_size );

    ::st_Buffer* eb = ::st_Buffer_new( requ_buffer_size );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    size_t   be_index = size_t{ 0 };
    belem_t* ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    object_t* ptr_object = ::st_Buffer_add_object( eb, ptr_orig, sizeof( belem_t ),
        BEAM_ELEMENT_TYPE_ID, ::st_DriftExact_get_num_dataptrs( ptr_orig ),
            nullptr, nullptr, nullptr );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
    ASSERT_TRUE( ::st_Object_get_const_begin_ptr( ptr_object ) != nullptr );
    ASSERT_TRUE( ::st_Object_get_size( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( ::st_Object_get_type_id( ptr_object ) == BEAM_ELEMENT_TYPE_ID );

    belem_t* ptr_drift = reinterpret_cast< belem_t* >(
        ::st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( EPS > std::fabs( ::st_DriftExact_get_length( ptr_drift ) -
                                  ::st_DriftExact_get_length( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = ::st_DriftExact_new( eb );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs(
        ::st_DriftExact_get_length( ptr_drift ) - ZERO ) );

    ::st_DriftExact_set_length( ptr_drift, ::st_DriftExact_get_length( ptr_orig ) );
    ASSERT_TRUE( EPS > std::fabs( ::st_DriftExact_get_length( ptr_drift ) -
                                  ::st_DriftExact_get_length( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = ::st_DriftExact_add( eb, ::st_DriftExact_get_length( ptr_orig ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
    ASSERT_TRUE( EPS > std::fabs( ::st_DriftExact_get_length( ptr_drift ) -
                                  ::st_DriftExact_get_length( ptr_orig  ) ) );

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig  = &orig_beam_elements[ be_index++ ];
        ptr_drift = ::st_DriftExact_add( eb, ::st_DriftExact_get_length( ptr_orig ) );

        ASSERT_TRUE( ptr_drift != nullptr );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
        ASSERT_TRUE( EPS > std::fabs( ::st_DriftExact_get_length( ptr_drift ) -
                                      ::st_DriftExact_get_length( ptr_orig  ) ) );
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

    be_index = size_t{ 0 };

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
    {
        ptr_orig = &orig_beam_elements[ be_index++ ];

        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) == BEAM_ELEMENT_TYPE_ID );
        ASSERT_TRUE( ::st_Object_get_type_id( obj_it ) ==
                     ::st_Object_get_type_id( cmp_it ) );

        ASSERT_TRUE( ::st_Object_get_size( obj_it ) >= sizeof( belem_t ) );
        ASSERT_TRUE( ::st_Object_get_size( obj_it ) ==
                     ::st_Object_get_size( cmp_it ) );

        belem_t const* elem = reinterpret_cast< belem_t const* >(
            ::st_Object_get_const_begin_ptr( obj_it ) );

        belem_t const* cmp_elem = reinterpret_cast< belem_t const* >(
            ::st_Object_get_const_begin_ptr( cmp_it ) );

        ASSERT_TRUE( ptr_orig != elem );
        ASSERT_TRUE( ptr_orig != cmp_elem );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( elem ) -
                                ::st_DriftExact_get_length( ptr_orig ) ) < EPS );

        ASSERT_TRUE( std::fabs( ::st_DriftExact_get_length( cmp_elem ) -
                                ::st_DriftExact_get_length( ptr_orig ) ) < EPS );
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_free( &cmp_buffer );
}

/* end: tests/sixtracklib/common/test_be_drift_c99.cpp  */
