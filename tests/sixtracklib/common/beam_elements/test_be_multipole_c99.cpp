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
#include "sixtracklib/common/be_multipole/be_multipole.h"

/* ************************************************************************* *
 * ****** NS(Multipole):
 * ************************************************************************* */

TEST( C99CommonBeamElementMultipole, MinimalAddToBufferCopyRemapRead )
{
    using    belem_t = NS(Multipole);
    using     size_t = NS(buffer_size_t);
    using   object_t = NS(Object);
    using      raw_t = unsigned char;
    using  mp_real_t = ::NS(multipole_real_t);
    using mp_order_t = ::NS(multipole_order_t);

    static double const ZERO = double{ 0.0 };
    static double const EPS  = std::numeric_limits< double >::epsilon();

    /* --------------------------------------------------------------------- */

    std::mt19937_64::result_type const seed = 20180830u;

    std::mt19937_64 prng;
    prng.seed( seed );

    using len_dist_t = std::uniform_real_distribution< mp_real_t >;
    using hxl_dist_t = std::uniform_real_distribution< mp_real_t >;
    using hyl_dist_t = std::uniform_real_distribution< mp_real_t >;
    using bal_dist_t = std::uniform_real_distribution< mp_real_t >;
    using ord_dist_t = std::uniform_int_distribution< mp_order_t >;

    len_dist_t length_dist( mp_real_t{   0.0 },  mp_real_t{ +10.0 } );
    hxl_dist_t hxl_dist(    mp_real_t{  -5.0 },  mp_real_t{  +5.0 } );
    hyl_dist_t hyl_dist(    mp_real_t{  -5.0 },  mp_real_t{  +5.0 } );
    bal_dist_t bal_dist(    mp_real_t{ -10.0 },  mp_real_t{ +10.0 } );
    ord_dist_t ord_dist(   mp_order_t{     0 }, mp_order_t{  20   } );

    static SIXTRL_CONSTEXPR_OR_CONST size_t
        NUM_BEAM_ELEMENTS = size_t{ 1000 };

    NS(object_type_id_t) const BEAM_ELEMENT_TYPE_ID =
        NS(OBJECT_TYPE_MULTIPOLE);

    std::vector< belem_t > orig_beam_elements( NUM_BEAM_ELEMENTS, belem_t{} );

    std::vector< mp_order_t > orig_multipole_orders(
        NUM_BEAM_ELEMENTS, mp_order_t{ 0 } );

    size_t const slot_size      = NS(BUFFER_DEFAULT_SLOT_SIZE);
    size_t const num_objs       = NUM_BEAM_ELEMENTS;
    size_t const num_garbage    = size_t{ 0 };

    size_t num_dataptrs         = size_t{ 0 };
    size_t num_slots            = size_t{ 0 };
    size_t num_bal_coefficients = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        mp_real_t  const length   = length_dist( prng );
        mp_real_t  const hxl      = hxl_dist( prng );
        mp_real_t  const hyl      = hyl_dist( prng );
        mp_order_t const order    = ord_dist( prng );

        belem_t* ptr_mp = NS(Multipole_preset)( &orig_beam_elements[ ii ] );
        ASSERT_TRUE( ptr_mp != nullptr );

        NS(Multipole_set_length)( ptr_mp, length );
        NS(Multipole_set_hxl)( ptr_mp, hxl );
        NS(Multipole_set_hyl)( ptr_mp, hyl );

        ASSERT_TRUE( std::fabs( length -
            NS(Multipole_length)( ptr_mp ) ) < EPS );

        ASSERT_TRUE( std::fabs( hxl - NS(Multipole_hxl)( ptr_mp ) ) < EPS );
        ASSERT_TRUE( std::fabs( hyl - NS(Multipole_hyl)( ptr_mp ) ) < EPS );

        if( order >= mp_order_t{ 0 } )
        {
            num_bal_coefficients += static_cast< size_t >( 2 * order + 1 );
        }

        orig_multipole_orders[ ii ] = order;
    }

    ASSERT_TRUE( num_bal_coefficients >= size_t{ 0 } );

    std::vector< mp_real_t > bal_coefficients;

    mp_real_t* bal_begin = bal_coefficients.data();
    mp_real_t* bal_buffer_end = bal_begin;

    if( num_bal_coefficients > size_t{ 0 } )
    {
        bal_coefficients.resize( num_bal_coefficients, mp_real_t{ 0.0 } );
        bal_begin = bal_coefficients.data();
        ASSERT_TRUE( bal_begin != nullptr );

        bal_buffer_end = bal_begin;
        std::advance( bal_buffer_end, bal_coefficients.size() );
    }

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        belem_t* ptr_mp = &orig_beam_elements[ ii ];
        mp_order_t const order = orig_multipole_orders[ ii ];

        if( order >= mp_order_t{ 0 } )
        {
            mp_real_t* bal_end = bal_begin;
            std::advance( bal_end, order );
            ASSERT_TRUE( std::distance( bal_end, bal_buffer_end ) >= 0 );

            mp_real_t* bal_it = bal_begin;

            for( ; bal_it != bal_end ; ++bal_it )
            {
                *bal_it = bal_dist( prng );
            }

            NS(arch_status_t) status = NS(Multipole_set_order)( ptr_mp, order );
            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
            ASSERT_TRUE( NS(Multipole_order)( ptr_mp ) == order );
            ASSERT_TRUE( NS(Multipole_bal_length)( ptr_mp ) ==
                            static_cast< size_t >( 2 * order + 2 ) );

            status = NS(Multipole_set_bal_addr)( ptr_mp,
                static_cast< NS(buffer_addr_t) >( reinterpret_cast<
                    uintptr_t >( bal_begin ) ) );
            ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
            mp_real_t const* cmp_bal_it = NS(Multipole_const_bal_begin)(
                ptr_mp );

            ASSERT_TRUE( cmp_bal_it != nullptr );

            bal_it = bal_begin;

            for( ; bal_it != bal_end ; ++bal_it, ++cmp_bal_it )
            {
                ASSERT_TRUE( std::fabs( *bal_it - *cmp_bal_it ) < EPS );
            }

            size_t const obj_num_dataptrs =
                NS(Multipole_num_dataptrs)( ptr_mp );

            size_t const sizes[]  = { sizeof( mp_real_t ) };
            size_t const counts[] = { NS(Multipole_bal_length)( ptr_mp ) };

            bal_begin = bal_end;

            num_slots += NS(ManagedBuffer_predict_required_num_slots)( nullptr,
                sizeof( belem_t ), obj_num_dataptrs, sizes, counts, slot_size );

            num_dataptrs += obj_num_dataptrs;
        }
        else
        {
            NS(Multipole_set_order)( ptr_mp, mp_order_t{ -1 } );
            num_slots += NS(ManagedBuffer_predict_required_num_slots)( nullptr,
                sizeof( belem_t ), NS(Multipole_num_dataptrs)( ptr_mp ),
                    nullptr, nullptr, slot_size );

        }

        ASSERT_TRUE( order == NS(Multipole_order)( ptr_mp ) );
    }

    /* --------------------------------------------------------------------- */

    belem_t const* ptr_orig   = nullptr;
    object_t*      ptr_object = nullptr;
    belem_t*       ptr_mp     = nullptr;

    size_t const requ_buffer_size = NS(ManagedBuffer_calculate_buffer_length)(
        nullptr, num_objs, num_slots, num_dataptrs, num_garbage, slot_size );

    ASSERT_TRUE( requ_buffer_size >=
                 NS(ManagedBuffer_get_header_length)( nullptr, slot_size ) );

    /* --------------------------------------------------------------------- */

    NS(Buffer)* eb = NS(Buffer_new)( requ_buffer_size );
    ASSERT_TRUE( eb != nullptr );

    /* --------------------------------------------------------------------- */

    size_t be_index = 0;

    ptr_orig = &orig_beam_elements[ be_index++ ];

    ASSERT_TRUE( ptr_orig != nullptr );
    ASSERT_TRUE( ( ( NS(Multipole_order)( ptr_orig ) >= mp_order_t{ 0 } ) &&
                   ( NS(Multipole_const_bal_begin)( ptr_orig ) != nullptr ) ) ||
                 ( ( NS(Multipole_order)( ptr_orig ) <  mp_order_t{ 0 } ) &&
                   ( NS(Multipole_const_bal_begin)( ptr_orig ) == nullptr ) ) );

    if( NS(Multipole_order)( ptr_orig )  >= mp_order_t{ 0 } )
    {
        size_t const offsets[] = { offsetof( belem_t, bal_addr ) };
        size_t const sizes[]   = { sizeof( ::NS(multipole_real_t) ) };
        size_t const counts[]  = { NS(Multipole_bal_length)( ptr_orig ) };

        ptr_object = NS(Buffer_add_object)( eb, ptr_orig, sizeof( belem_t ),
            BEAM_ELEMENT_TYPE_ID, NS(Multipole_num_dataptrs)( ptr_orig ),
                offsets, sizes, counts );
    }
    else
    {
        ptr_object = NS(Buffer_add_object)( eb, ptr_orig, sizeof( belem_t ),
            BEAM_ELEMENT_TYPE_ID, NS(Multipole_num_dataptrs)( ptr_orig ),
                nullptr, nullptr, nullptr );
    }

    ASSERT_TRUE( ptr_object != nullptr );

    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) == be_index );
    ASSERT_TRUE( NS(Object_get_const_begin_ptr)( ptr_object ) != nullptr );
    ASSERT_TRUE( NS(Object_get_size)( ptr_object ) >= sizeof( belem_t ) );
    ASSERT_TRUE( NS(Object_get_type_id)( ptr_object ) == BEAM_ELEMENT_TYPE_ID );

    ptr_mp = reinterpret_cast< belem_t* >(
        NS(Object_get_begin_ptr)( ptr_object ) );

    ASSERT_TRUE( ptr_mp != nullptr );

    ASSERT_TRUE( std::fabs( NS(Multipole_length)( ptr_mp ) -
                            NS(Multipole_length)( ptr_orig ) ) < EPS );

    ASSERT_TRUE( std::fabs( NS(Multipole_hxl)( ptr_mp ) -
                            NS(Multipole_hxl)( ptr_orig ) ) < EPS );

    ASSERT_TRUE( std::fabs( NS(Multipole_hyl)( ptr_mp ) -
                            NS(Multipole_hyl)( ptr_orig ) ) < EPS );

    ASSERT_TRUE( NS(Multipole_order)( ptr_mp ) ==
                 NS(Multipole_order)( ptr_orig ) );

    if( NS(Multipole_order)( ptr_orig ) > mp_order_t{ 0 } )
    {
        mp_real_t const* orig_bal_it  = NS(Multipole_const_bal_begin)( ptr_orig );
        mp_real_t const* orig_bal_end = orig_bal_it;
        std::advance( orig_bal_end, NS(Multipole_bal_length)( ptr_orig ) );

        mp_real_t const* cmp_bal_it   = NS(Multipole_const_bal_begin)( ptr_mp );

        for( ; orig_bal_it != orig_bal_end ; ++orig_bal_it, ++cmp_bal_it )
        {
            ASSERT_TRUE( std::fabs( *orig_bal_it - *cmp_bal_it ) < EPS );
        }
    }
    else
    {
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_orig ) == nullptr );
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp   ) == nullptr );
    }

    /* --------------------------------------------------------------------- */

    ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    ptr_mp = NS(Multipole_new)( eb, NS(Multipole_order)( ptr_orig ) );
    ASSERT_TRUE( ptr_mp != nullptr );

    ASSERT_TRUE( std::fabs( NS(Multipole_length)( ptr_mp ) - ZERO ) < EPS );
    ASSERT_TRUE( std::fabs( NS(Multipole_hxl)( ptr_mp )    - ZERO ) < EPS );
    ASSERT_TRUE( std::fabs( NS(Multipole_hyl)( ptr_mp )    - ZERO ) < EPS );

    NS(Multipole_set_length)( ptr_mp, NS(Multipole_length)( ptr_orig ) );
    NS(Multipole_set_hxl)(    ptr_mp, NS(Multipole_hxl)( ptr_orig ) );
    NS(Multipole_set_hyl)(    ptr_mp, NS(Multipole_hyl)( ptr_orig ) );

    ASSERT_TRUE( NS(Multipole_order)( ptr_mp ) ==
                 NS(Multipole_order)( ptr_orig ) );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_length)( ptr_orig ) -
                                  NS(Multipole_length)( ptr_mp ) ) );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hxl)( ptr_orig ) -
                                  NS(Multipole_hxl)( ptr_mp ) ) );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hyl)( ptr_orig ) -
                                  NS(Multipole_hyl)( ptr_mp ) ) );

    if( mp_order_t{ 0 } <= NS(Multipole_order)( ptr_orig ) )
    {
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) != nullptr );
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) !=
                     NS(Multipole_const_bal_begin)( ptr_orig ) );

        size_t const bal_size = NS(Multipole_bal_length)( ptr_orig );

        for( size_t ii = size_t{ 0 } ; ii < bal_size ; ++ii )
        {
            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp, ii ) - ZERO ) );
        }

        NS(Multipole_set_bal)( ptr_mp,
            NS(Multipole_const_bal_begin)( ptr_orig ) );

        for( mp_order_t ii = NS(Multipole_order)( ptr_mp ) ;
             ii >= mp_order_t{ 0 } ; --ii )
        {
            SIXTRL_ASSERT( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp,   2u * ii      ) -
                NS(Multipole_bal)( ptr_orig, 2u * ii      ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp,   2u * ii      ) -
                NS(Multipole_bal)( ptr_orig, 2u * ii      ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp,   2u * ii + 1u ) -
                NS(Multipole_bal)( ptr_orig, 2u * ii + 1u ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_knl)( ptr_mp,   ii ) -
                NS(Multipole_knl)( ptr_orig, ii ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_ksl)( ptr_mp,   ii ) -
                NS(Multipole_ksl)( ptr_orig, ii ) ) );
        }
    }
    else
    {
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp )   == nullptr );
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_orig ) == nullptr );
    }

    /* --------------------------------------------------------------------- */

    ptr_orig = &orig_beam_elements[ be_index++ ];
    ASSERT_TRUE( ptr_orig != nullptr );

    ptr_mp   = NS(Multipole_add)( eb,
        NS(Multipole_order)( ptr_orig ),
        NS(Multipole_length)( ptr_orig ),
        NS(Multipole_hxl)( ptr_orig ),
        NS(Multipole_hyl)( ptr_orig ),
        reinterpret_cast< uintptr_t >(
            NS(Multipole_const_bal_begin)( ptr_orig ) ) );

    ASSERT_TRUE( ptr_mp != nullptr );

    mp_order_t order = NS(Multipole_order)( ptr_orig );

    ASSERT_TRUE( NS(Multipole_order)( ptr_mp ) == order );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_length)( ptr_orig ) -
                                  NS(Multipole_length)( ptr_mp ) ) );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hxl)( ptr_orig ) -
                                  NS(Multipole_hxl)( ptr_mp ) ) );

    ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hyl)( ptr_orig ) -
                                  NS(Multipole_hyl)( ptr_mp ) ) );

    if( mp_order_t{ 0 } <= order )
    {
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) != nullptr );
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) !=
                     NS(Multipole_const_bal_begin)( ptr_orig ) );

        for( mp_order_t ii = order ; ii >= mp_order_t{ 0 } ; --ii )
        {
            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp,   2u * ii      ) -
                NS(Multipole_bal)( ptr_orig, 2u * ii      ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_bal)( ptr_mp,   2u * ii + 1u ) -
                NS(Multipole_bal)( ptr_orig, 2u * ii + 1u ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_knl)( ptr_mp,   ii ) -
                NS(Multipole_knl)( ptr_orig, ii ) ) );

            ASSERT_TRUE( EPS > std::fabs(
                NS(Multipole_ksl)( ptr_mp,   ii ) -
                NS(Multipole_ksl)( ptr_orig, ii ) ) );
        }
    }
    else
    {
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp )   == nullptr );
        ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_orig ) == nullptr );
    }

    /* --------------------------------------------------------------------- */

    for( ; be_index < NUM_BEAM_ELEMENTS ; )
    {
        ptr_orig = &orig_beam_elements[ be_index++ ];
        ASSERT_TRUE( ptr_orig != nullptr );

        ptr_mp   = NS(Multipole_add)( eb,
            NS(Multipole_order)( ptr_orig ),
            NS(Multipole_length)( ptr_orig ),
            NS(Multipole_hxl)( ptr_orig ),
            NS(Multipole_hyl)( ptr_orig ),
            reinterpret_cast< uintptr_t >(
                NS(Multipole_const_bal_begin)( ptr_orig ) ) );

        ASSERT_TRUE( ptr_mp != nullptr );

        mp_order_t order = NS(Multipole_order)( ptr_orig );

        ASSERT_TRUE( NS(Multipole_order)( ptr_mp ) == order );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_length)( ptr_orig ) -
                                      NS(Multipole_length)( ptr_mp ) ) );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hxl)( ptr_orig ) -
                                      NS(Multipole_hxl)( ptr_mp ) ) );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hyl)( ptr_orig ) -
                                      NS(Multipole_hyl)( ptr_mp ) ) );

        if( mp_order_t{ 0 } <= order )
        {
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) != nullptr );
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp ) !=
                        NS(Multipole_const_bal_begin)( ptr_orig ) );

            for( mp_order_t ii = order ; ii >= mp_order_t{ 0 } ; --ii )
            {
                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_bal)( ptr_mp,   2u * ii      ) -
                    NS(Multipole_bal)( ptr_orig, 2u * ii      ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_bal)( ptr_mp,   2u * ii + 1u ) -
                    NS(Multipole_bal)( ptr_orig, 2u * ii + 1u ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_knl)( ptr_mp,   ii ) -
                    NS(Multipole_knl)( ptr_orig, ii ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_ksl)( ptr_mp,   ii ) -
                    NS(Multipole_ksl)( ptr_orig, ii ) ) );
            }
        }
        else
        {
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_mp )   == nullptr );
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( ptr_orig ) == nullptr );
        }
    }

    /* ===================================================================== */

    ASSERT_TRUE( NS(Buffer_get_size)( eb ) > size_t{ 0 } );

    std::vector< raw_t > copy_buffer( NS(Buffer_get_size)( eb ), raw_t{ 0 } );
    copy_buffer.assign( NS(Buffer_get_const_data_begin)( eb ),
                        NS(Buffer_get_const_data_end)( eb ) );

    NS(Buffer) cmp_buffer;
    NS(Buffer_preset)( &cmp_buffer );

    int success = NS(Buffer_init)(
        &cmp_buffer, copy_buffer.data(), copy_buffer.size() );

    ASSERT_TRUE( success == 0 );
    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) ==
                 NS(Buffer_get_num_of_objects)( &cmp_buffer ) );

    ASSERT_TRUE( NS(Buffer_get_num_of_objects)( eb ) ==
                 orig_beam_elements.size() );

    object_t const* obj_it  = NS(Buffer_get_const_objects_begin)( eb );
    object_t const* obj_end = NS(Buffer_get_const_objects_end)( eb );
    object_t const* cmp_it  = NS(Buffer_get_const_objects_begin)( &cmp_buffer );

    ptr_orig = &orig_beam_elements[ 0 ];

    for( ; obj_it != obj_end ; ++obj_it, ++cmp_it, ++ptr_orig )
    {
        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) == BEAM_ELEMENT_TYPE_ID );
        ASSERT_TRUE( NS(Object_get_type_id)( obj_it ) ==
                     NS(Object_get_type_id)( cmp_it ) );

        ASSERT_TRUE( NS(Object_get_size)( obj_it ) >= sizeof( belem_t ) );
        ASSERT_TRUE( NS(Object_get_size)( obj_it ) ==
                     NS(Object_get_size)( cmp_it ) );


        belem_t const* elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( obj_it ) );

        belem_t const* cmp_elem = reinterpret_cast< belem_t const* >(
            NS(Object_get_const_begin_ptr)( cmp_it ) );


        ASSERT_TRUE( elem     != nullptr );
        ASSERT_TRUE( cmp_elem != nullptr );
        ASSERT_TRUE( cmp_elem != elem    );

        mp_order_t order = NS(Multipole_order)( elem );

        ASSERT_TRUE( NS(Multipole_order)( cmp_elem ) == order );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_length)( elem ) -
                                      NS(Multipole_length)( cmp_elem ) ) );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hxl)( elem ) -
                                      NS(Multipole_hxl)( cmp_elem ) ) );

        ASSERT_TRUE( EPS > std::fabs( NS(Multipole_hyl)( elem ) -
                                      NS(Multipole_hyl)( cmp_elem ) ) );

        if( mp_order_t{ 0 } <= order )
        {
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( cmp_elem ) != nullptr );
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( cmp_elem ) !=
                         NS(Multipole_const_bal_begin)( elem ) );

            for( mp_order_t ii = order ; ii >= mp_order_t{ 0 } ; --ii )
            {
                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_bal)( cmp_elem, 2u * ii      ) -
                    NS(Multipole_bal)( elem, 2u * ii      ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_bal)( cmp_elem, 2u * ii + 1u ) -
                    NS(Multipole_bal)( elem, 2u * ii + 1u ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_knl)( cmp_elem, ii ) -
                    NS(Multipole_knl)( elem, ii ) ) );

                ASSERT_TRUE( EPS > std::fabs(
                    NS(Multipole_ksl)( cmp_elem, ii ) -
                    NS(Multipole_ksl)( elem, ii ) ) );
            }
        }
        else
        {
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( cmp_elem ) == nullptr );
            ASSERT_TRUE( NS(Multipole_const_bal_begin)( elem ) == nullptr );
        }
    }

    /* ===================================================================== */

    NS(Buffer_delete)( eb );
    NS(Buffer_free)( &cmp_buffer );
}
