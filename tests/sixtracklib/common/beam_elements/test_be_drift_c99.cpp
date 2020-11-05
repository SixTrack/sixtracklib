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
#include "sixtracklib/common/be_drift/be_drift.h"

/* ************************************************************************* *
 * ****** NS(Drift):
 * ************************************************************************* */

TEST( C99CommonBeamElementDrift, MinimalAddToBufferCopyRemapRead )
{
    using size_t   = NS(buffer_size_t);
    using object_t = NS(Object);
    using raw_t    = unsigned char;
    using belem_t  = NS(Drift);
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

    NS(object_type_id_t) const BELEM_TYPE_ID = NS(OBJECT_TYPE_DRIFT);
    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    size_t const slot_size      = NS(BUFFER_DEFAULT_SLOT_SIZE);
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };
    size_t const num_dataptrs   = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        real_t const length = length_dist( prng );

        belem_t* ptr_drift = NS(Drift_preset)( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_drift != nullptr );
        NS(Drift_set_length)( ptr_drift, length );

        ASSERT_TRUE( std::fabs( length - NS(Drift_length)( ptr_drift ) ) < EPS );

        num_slots += NS(ManagedBuffer_predict_required_num_slots)( nullptr,
            sizeof( NS(Drift) ), NS(Drift_num_dataptrs)( ptr_drift ),
                nullptr, nullptr, slot_size );
    }

    /* --------------------------------------------------------------------- */

    size_t const requ_buffer_size = NS(ManagedBuffer_calculate_buffer_length)(
        nullptr, num_objs, num_slots, num_dataptrs, num_garbage, slot_size );

    NS(Buffer)* eb = NS(Buffer_new)( requ_buffer_size );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    size_t   be_index = size_t{ 0 };
    belem_t* ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    object_t* ptr_object = NS(Buffer_add_object)( eb, ptr_orig, sizeof( belem_t ),
        BELEM_TYPE_ID, NS(Drift_num_dataptrs)( ptr_orig ),
            nullptr, nullptr, nullptr );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
    ASSERT_TRUE( NS(Object_get_const_begin_ptr)( ptr_object ) != nullptr );
    ASSERT_TRUE( NS(Object_get_size)( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( NS(Object_get_type_id)( ptr_object ) == BELEM_TYPE_ID );

    belem_t* ptr_drift = reinterpret_cast< belem_t* >(
        NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( EPS > std::fabs( NS(Drift_length)( ptr_drift ) -
                                  NS(Drift_length)( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = NS(Drift_new)( eb );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs( NS(Drift_length)( ptr_drift ) - ZERO ) );

    NS(Drift_set_length)( ptr_drift, NS(Drift_length)( ptr_orig ) );
    ASSERT_TRUE( EPS > std::fabs( NS(Drift_length)( ptr_drift ) -
                                  NS(Drift_length)( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = NS(Drift_add)( eb, NS(Drift_length)( ptr_orig ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
    ASSERT_TRUE( EPS > std::fabs( NS(Drift_length)( ptr_drift ) -
                                  NS(Drift_length)( ptr_orig  ) ) );

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig  = &orig_beam_elements[ be_index++ ];
        ptr_drift = NS(Drift_add)( eb, NS(Drift_length)( ptr_orig ) );

        ASSERT_TRUE( ptr_drift != nullptr );
        ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
        ASSERT_TRUE( EPS > std::fabs( NS(Drift_length)( ptr_drift ) -
                                      NS(Drift_length)( ptr_orig  ) ) );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( NS(Buffer_get_size)( eb ) > size_t{ 0 } );

    std::vector< raw_t > data_buffer( NS(Buffer_get_size)( eb ), raw_t{ 0 } );
    data_buffer.assign( NS(Buffer_get_const_data_begin)( eb ),
                        NS(Buffer_get_const_data_end)( eb ) );

    NS(Buffer) cmp_buffer;
    NS(Buffer_preset)( &cmp_buffer );
    int success = NS(Buffer_init)(
        &cmp_buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) ==
                 NS(Buffer_get_num_of_objects)( &cmp_buffer ) );

    object_t const* obj_it  = NS(Buffer_get_const_objects_begin)( eb );
    object_t const* obj_end = NS(Buffer_get_const_objects_end)( eb );
    object_t const* cmp_it  = NS(Buffer_get_const_objects_begin)( &cmp_buffer );

    be_index = size_t{ 0 };

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
    {
        ptr_orig = &orig_beam_elements[ be_index++ ];

        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) == BELEM_TYPE_ID );
        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) ==
                     NS(Object_get_type_id)( cmp_it ) );

        ASSERT_TRUE( NS(Object_get_size)( obj_it ) >= sizeof( belem_t ) );
        ASSERT_TRUE( NS(Object_get_size)( obj_it ) ==
                     NS(Object_get_size)( cmp_it ) );

        belem_t const* elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( obj_it ) );

        belem_t const* cmp_elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( cmp_it ) );

        ASSERT_TRUE( ptr_orig != elem );
        ASSERT_TRUE( ptr_orig != cmp_elem );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( NS(Drift_length)( elem ) -
                                NS(Drift_length)( ptr_orig ) ) < EPS );

        ASSERT_TRUE( std::fabs( NS(Drift_length)( cmp_elem ) -
                                NS(Drift_length)( ptr_orig ) ) < EPS );
    }

    /* --------------------------------------------------------------------- */

    NS(Buffer_delete)( eb );
    NS(Buffer_free)( &cmp_buffer );
}

/* ************************************************************************* *
 * ****** NS(DriftExact):
 * ************************************************************************* */

TEST( C99CommonBeamElementDriftExact, MinimalAddToBufferCopyRemapRead )
{
    using size_t   = NS(buffer_size_t);
    using object_t = NS(Object);
    using raw_t    = unsigned char;
    using belem_t  = NS(DriftExact);
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

    NS(object_type_id_t) const BELEM_TYPE_ID = NS(OBJECT_TYPE_DRIFT_EXACT);
    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    size_t const slot_size      = NS(BUFFER_DEFAULT_SLOT_SIZE);
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };
    size_t const num_dataptrs   = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        real_t  const length   = length_dist( prng );

        belem_t* ptr_drift = NS(DriftExact_preset)( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_drift != nullptr );
        NS(DriftExact_set_length)( ptr_drift, length );

        ASSERT_TRUE( EPS > std::fabs( length -
            NS(DriftExact_length)( ptr_drift ) ) );

        num_slots += NS(ManagedBuffer_predict_required_num_slots)( nullptr,
            sizeof( NS(DriftExact) ),
                NS(DriftExact_num_dataptrs)( ptr_drift ), nullptr,
                    nullptr, slot_size );
    }

    /* --------------------------------------------------------------------- */

    size_t const requ_buffer_size = NS(ManagedBuffer_calculate_buffer_length)(
        nullptr, num_objs, num_slots, num_dataptrs, num_garbage, slot_size );

    NS(Buffer)* eb = NS(Buffer_new)( requ_buffer_size );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    size_t   be_index = size_t{ 0 };
    belem_t* ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    object_t* ptr_object = NS(Buffer_add_object)( eb, ptr_orig, sizeof( belem_t ),
        BELEM_TYPE_ID, NS(DriftExact_num_dataptrs)( ptr_orig ),
            nullptr, nullptr, nullptr );

    ASSERT_TRUE( ptr_object != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
    ASSERT_TRUE( NS(Object_get_const_begin_ptr)( ptr_object ) != nullptr );
    ASSERT_TRUE( NS(Object_get_size)( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( NS(Object_get_type_id)( ptr_object ) == BELEM_TYPE_ID );

    belem_t* ptr_drift = reinterpret_cast< belem_t* >(
        NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( EPS > std::fabs( NS(DriftExact_length)( ptr_drift ) -
                                  NS(DriftExact_length)( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = NS(DriftExact_new)( eb );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );

    ASSERT_TRUE( EPS > std::fabs(
        NS(DriftExact_length)( ptr_drift ) - ZERO ) );

    NS(DriftExact_set_length)( ptr_drift, NS(DriftExact_length)( ptr_orig ) );
    ASSERT_TRUE( EPS > std::fabs( NS(DriftExact_length)( ptr_drift ) -
                                  NS(DriftExact_length)( ptr_orig  ) ) );

    /* --------------------------------------------------------------------- */

    ptr_orig  = &orig_beam_elements[ be_index++ ];
    ptr_drift = NS(DriftExact_add)( eb, NS(DriftExact_length)( ptr_orig ) );

    ASSERT_TRUE( ptr_drift != nullptr );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
    ASSERT_TRUE( EPS > std::fabs( NS(DriftExact_length)( ptr_drift ) -
                                  NS(DriftExact_length)( ptr_orig  ) ) );

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig  = &orig_beam_elements[ be_index++ ];
        ptr_drift = NS(DriftExact_add)( eb, NS(DriftExact_length)( ptr_orig ) );

        ASSERT_TRUE( ptr_drift != nullptr );
        ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
        ASSERT_TRUE( EPS > std::fabs( NS(DriftExact_length)( ptr_drift ) -
                                      NS(DriftExact_length)( ptr_orig  ) ) );
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( NS(Buffer_get_size)( eb ) > size_t{ 0 } );

    std::vector< raw_t > data_buffer( NS(Buffer_get_size)( eb ), raw_t{ 0 } );
    data_buffer.assign( NS(Buffer_get_const_data_begin)( eb ),
                        NS(Buffer_get_const_data_end)( eb ) );

    NS(Buffer) cmp_buffer;
    NS(Buffer_preset)( &cmp_buffer );
    int success = NS(Buffer_init)(
        &cmp_buffer, data_buffer.data(), data_buffer.size() );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) ==
                 NS(Buffer_get_num_of_objects)( &cmp_buffer ) );

    object_t const* obj_it  = NS(Buffer_get_const_objects_begin)( eb );
    object_t const* obj_end = NS(Buffer_get_const_objects_end)( eb );
    object_t const* cmp_it  = NS(Buffer_get_const_objects_begin)( &cmp_buffer );

    be_index = size_t{ 0 };

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
    {
        ptr_orig = &orig_beam_elements[ be_index++ ];

        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) == BELEM_TYPE_ID );
        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) ==
                     NS(Object_get_type_id)( cmp_it ) );

        ASSERT_TRUE( NS(Object_get_size)( obj_it ) >= sizeof( belem_t ) );
        ASSERT_TRUE( NS(Object_get_size)( obj_it ) ==
                     NS(Object_get_size)( cmp_it ) );

        belem_t const* elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( obj_it ) );

        belem_t const* cmp_elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( cmp_it ) );

        ASSERT_TRUE( ptr_orig != elem );
        ASSERT_TRUE( ptr_orig != cmp_elem );

        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        ASSERT_TRUE( std::fabs( NS(DriftExact_length)( elem ) -
                                NS(DriftExact_length)( ptr_orig ) ) < EPS );

        ASSERT_TRUE( std::fabs( NS(DriftExact_length)( cmp_elem ) -
                                NS(DriftExact_length)( ptr_orig ) ) < EPS );
    }

    /* --------------------------------------------------------------------- */

    NS(Buffer_delete)( eb );
    NS(Buffer_free)( &cmp_buffer );
}
