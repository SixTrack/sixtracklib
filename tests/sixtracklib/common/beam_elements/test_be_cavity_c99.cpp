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

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"

#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_cavity/be_cavity.h"

/* ************************************************************************* *
 * ******  st_Cavity:
 * ************************************************************************* */

TEST( C99_CommonBeamElementCavityTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t   = ::st_buffer_size_t;
    using object_t = ::st_Object;
    using raw_t    = unsigned char;
    using belem_t  = ::st_Cavity;
    using real_t   = SIXTRL_REAL_T;

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    /* --------------------------------------------------------------------- */

    std::mt19937_64::result_type const seed = 20180830u;

    std::mt19937_64 prng;
    prng.seed( seed );

    using voltage_dist_t   = std::uniform_real_distribution< real_t >;
    using frequency_dist_t = std::uniform_real_distribution< real_t >;
    using lag_dist_t       = std::uniform_real_distribution< real_t >;

    voltage_dist_t      voltage_dist(   real_t{  0.5 }, real_t{ 1e5 } );
    frequency_dist_t    frequency_dist( real_t{ 50.0 }, real_t{ 200e3 } );
    lag_dist_t          lag_dist( real_t{ 0.0 }, real_t{ 10.0 } );

    static SIXTRL_CONSTEXPR_OR_CONST size_t
        NUM_BEAM_ELEMENTS = size_t{ 1000 };

    ::st_object_type_id_t const BEAM_ELEMENT_TYPE_ID = ::st_OBJECT_TYPE_CAVITY;
    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    size_t const slot_size      = ::st_BUFFER_DEFAULT_SLOT_SIZE;
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };
    size_t const num_dataptrs   = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        real_t const voltage   = voltage_dist( prng );
        real_t const frequency = frequency_dist( prng );
        real_t const lag       = lag_dist( prng );

        belem_t* ptr_cav = ::st_Cavity_preset( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_cav != nullptr );
        ::st_Cavity_set_voltage( ptr_cav, voltage );
        ::st_Cavity_set_frequency( ptr_cav, frequency );
        ::st_Cavity_set_lag( ptr_cav, lag );

        ASSERT_TRUE( EPS > std::fabs(
            voltage  - ::st_Cavity_get_voltage( ptr_cav ) ) );

        ASSERT_TRUE( EPS > std::fabs(
            frequency - ::st_Cavity_get_frequency( ptr_cav ) ) );

        ASSERT_TRUE( EPS > std::fabs( lag - ::st_Cavity_get_lag( ptr_cav ) ) );

        num_slots += ::st_ManagedBuffer_predict_required_num_slots( nullptr,
            sizeof( ::st_Cavity ), ::st_Cavity_get_num_dataptrs( ptr_cav ),
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
        BEAM_ELEMENT_TYPE_ID, ::st_Cavity_get_num_dataptrs( ptr_orig ),
            nullptr, nullptr, nullptr );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );
    ASSERT_TRUE( ::st_Object_get_const_begin_ptr( ptr_object ) != nullptr );
    ASSERT_TRUE( ::st_Object_get_size( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( ::st_Object_get_type_id( ptr_object ) == BEAM_ELEMENT_TYPE_ID );

    belem_t* ptr_cav = reinterpret_cast< belem_t* >(
        ::st_Object_get_begin_ptr( ptr_object ) );

    ASSERT_TRUE( ptr_cav != nullptr );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( ptr_cav ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( ptr_cav ) -
                                  ::st_Cavity_get_frequency( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( ptr_cav ) -
                                  ::st_Cavity_get_lag( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_cav = ::st_Cavity_new( eb );

    ASSERT_TRUE( ptr_cav != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( ptr_cav ) - ZERO ) );

    ::st_Cavity_set_voltage( ptr_cav, ::st_Cavity_get_voltage( ptr_orig ) );
    ::st_Cavity_set_frequency( ptr_cav, ::st_Cavity_get_frequency( ptr_orig ) );
    ::st_Cavity_set_lag( ptr_cav, ::st_Cavity_get_lag( ptr_orig ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( ptr_cav ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( ptr_cav ) -
                                  ::st_Cavity_get_frequency( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( ptr_cav ) -
                                  ::st_Cavity_get_lag( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_cav = ::st_Cavity_add( eb,
        ::st_Cavity_get_voltage( ptr_orig ),
        ::st_Cavity_get_frequency( ptr_orig ),
        ::st_Cavity_get_lag( ptr_orig ) );

    ASSERT_TRUE( ptr_cav != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( ptr_cav ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( ptr_cav ) -
                                  ::st_Cavity_get_frequency( ptr_orig  ) ) );

    ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( ptr_cav ) -
                                  ::st_Cavity_get_lag( ptr_orig  ) ) );

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig  = &orig_beam_elements[ be_index++ ];
        ptr_cav = ::st_Cavity_add( eb,
            ::st_Cavity_get_voltage( ptr_orig ),
            ::st_Cavity_get_frequency( ptr_orig ),
            ::st_Cavity_get_lag( ptr_orig ) );

        ASSERT_TRUE( ptr_cav != nullptr );
        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == be_index );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( ptr_cav ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( ptr_cav ) -
                                    ::st_Cavity_get_frequency( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( ptr_cav ) -
                                    ::st_Cavity_get_lag( ptr_orig  ) ) );
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

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( elem ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( elem ) -
                                    ::st_Cavity_get_frequency( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( elem ) -
                                    ::st_Cavity_get_lag( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_voltage( cmp_elem ) -
                                  ::st_Cavity_get_voltage( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_frequency( cmp_elem ) -
                                    ::st_Cavity_get_frequency( ptr_orig  ) ) );

        ASSERT_TRUE( EPS > std::fabs( ::st_Cavity_get_lag( cmp_elem ) -
                                    ::st_Cavity_get_lag( ptr_orig  ) ) );
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_free( &cmp_buffer );
}

/* end: tests/sixtracklib/common/test_be_cavity_c99.cpp */
