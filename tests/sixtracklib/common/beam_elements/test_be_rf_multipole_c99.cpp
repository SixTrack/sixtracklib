#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "sixtracklib/testlib.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"



TEST( C99CommonBeamElementsRFMultiPoleTests, StoreAndRestoreTests )
{
    using rf_multipole_t = ::NS(RFMultiPole);
    using rf_mp_int_t = ::NS(rf_multipole_int_t);
    using rf_mp_real_t = ::NS(rf_multipole_real_t);
    using buf_size_t = ::NS(buffer_size_t);
    using buffer_t = ::NS(Buffer);
    using real_limit_t = std::numeric_limits< rf_mp_real_t >;

    static const rf_mp_real_t REAL_EPS = real_limit_t::epsilon();
    static const rf_mp_int_t rfmp_01_order = rf_mp_int_t{ 1 };

    buffer_t* lattice = NS(Buffer_new)( buf_size_t{ 0 } );

    buf_size_t const rfmp_01_idx =
        ::NS(Buffer_get_num_of_objects)( lattice );

    rf_multipole_t* rfmp_01 = ::NS(RFMultiPole_new)( lattice, rfmp_01_order );

    ASSERT_TRUE( rfmp_01_idx < ::NS(Buffer_get_num_of_objects)( lattice ) );
    ASSERT_TRUE( rfmp_01 != nullptr );
    ASSERT_TRUE( ::NS(RFMultiPole_order)( rfmp_01 ) == rfmp_01_order );

    ASSERT_TRUE( ::NS(RFMultiPole_num_bal_elements)( rfmp_01 ) ==
        static_cast< rf_mp_int_t >( 2 * rfmp_01_order + 2 ) );

    ASSERT_TRUE( ::NS(RFMultiPole_num_phase_elements)( rfmp_01 ) ==
        static_cast< rf_mp_int_t >( 2 * rfmp_01_order + 2 ) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_voltage)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_frequency)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_lag)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( ::NS(RFMultiPole_const_bal)( rfmp_01 ) != nullptr );
    ASSERT_TRUE( ::NS(RFMultiPole_const_phase)( rfmp_01 ) != nullptr );

    /* --------------------------------------------------------------------- */

    rf_mp_real_t const bal_values[] =
    {
        rf_mp_real_t{ 0.0 }, rf_mp_real_t{ 0.0 },
        rf_mp_real_t{ 1.0 }, rf_mp_real_t{ 1.0 },
        rf_mp_real_t{ 2.0 }, rf_mp_real_t{ 2.0 }
    };

    rf_mp_int_t const rfmp_02_order = rf_mp_int_t{ 2 };
    rf_mp_real_t const rfmp_02_voltage = rf_mp_real_t{ 10e6 };
    rf_mp_real_t const rfmp_02_frequency = rf_mp_real_t{ 440e3 };
    rf_mp_real_t const rfmp_02_lag = rf_mp_real_t{ 0.0 };

    buf_size_t const rfmp_02_idx = ::NS(Buffer_get_num_of_objects)( lattice );

    rf_multipole_t* rfmp_02 = ::NS(RFMultiPole_add)( lattice, rfmp_02_order,
        rfmp_02_voltage, rfmp_02_frequency, rfmp_02_lag, &bal_values[ 0 ],
            nullptr );

    ASSERT_TRUE( rfmp_02_idx < ::NS(Buffer_get_num_of_objects)( lattice ) );
    ASSERT_TRUE( rfmp_02 != nullptr );
    ASSERT_TRUE( ::NS(RFMultiPole_order)( rfmp_02 ) == rfmp_02_order );

    ASSERT_TRUE( ::NS(RFMultiPole_num_bal_elements)( rfmp_02 ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultiPole_num_phase_elements)( rfmp_02 ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_voltage)(
        rfmp_02 ) - rfmp_02_voltage ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_frequency)(
        rfmp_02 ) - rfmp_02_frequency ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_lag)(
        rfmp_02 ) - rfmp_02_lag ) < REAL_EPS );

    ASSERT_TRUE( ::NS(RFMultiPole_const_bal)( rfmp_02 ) != nullptr );
    ASSERT_TRUE( ::NS(RFMultiPole_const_phase)( rfmp_02 ) != nullptr );

    ASSERT_TRUE( std::memcmp( &bal_values[ 0 ], ::NS(RFMultiPole_const_bal)(
        rfmp_02 ), sizeof( rf_mp_real_t ) * ::NS(RFMultiPole_num_bal_elements)(
            rfmp_02 ) ) == 0 );

    /* --------------------------------------------------------------------- */

    buf_size_t const rfmp_02_copy_idx =
        ::NS(Buffer_get_num_of_objects)( lattice );

    rf_multipole_t* rfmp_02_copy =
        ::NS(RFMultiPole_new)( lattice, rfmp_02_order );

    rfmp_02 = ::NS(RFMultiPole_from_buffer)( lattice, rfmp_02_idx );

    ASSERT_TRUE( rfmp_02_copy_idx > rfmp_02_idx );
    ASSERT_TRUE( rfmp_02_copy_idx <
                    ::NS(Buffer_get_num_of_objects)( lattice ) );

    ASSERT_TRUE( rfmp_02 != nullptr );
    ASSERT_TRUE( rfmp_02_copy != nullptr );
    ASSERT_TRUE( rfmp_02_copy != rfmp_02 );
    ASSERT_TRUE( ::NS(RFMultiPole_order)( rfmp_02 ) ==
                 ::NS(RFMultiPole_order)( rfmp_02_copy ) );

    ASSERT_TRUE( ::NS(RFMultiPole_num_bal_elements)( rfmp_02_copy ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultiPole_num_phase_elements)( rfmp_02_copy ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultiPole_const_bal)( rfmp_02_copy ) != nullptr );
    ASSERT_TRUE( ::NS(RFMultiPole_const_phase)( rfmp_02_copy ) != nullptr );

    ASSERT_TRUE( ::NS(RFMultiPole_copy)( rfmp_02_copy, rfmp_02 ) ==
                 ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_voltage)(
        rfmp_02_copy ) - rfmp_02_voltage ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_frequency)(
        rfmp_02_copy ) - rfmp_02_frequency ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultiPole_lag)(
        rfmp_02_copy ) - rfmp_02_lag ) < REAL_EPS );

    ASSERT_TRUE( std::memcmp( &bal_values[ 0 ], ::NS(RFMultiPole_const_bal)(
        rfmp_02_copy ), sizeof( rf_mp_real_t ) *
            ::NS(RFMultiPole_num_bal_elements)( rfmp_02_copy ) ) == 0 );

    ASSERT_TRUE( std::memcmp( ::NS(RFMultiPole_const_bal)( rfmp_02 ),
        ::NS(RFMultiPole_const_bal)( rfmp_02_copy ), sizeof( rf_mp_real_t ) *
            ::NS(RFMultiPole_num_bal_elements)( rfmp_02 ) ) == 0 );

    ASSERT_TRUE( std::memcmp( ::NS(RFMultiPole_const_phase)( rfmp_02 ),
        ::NS(RFMultiPole_const_phase)( rfmp_02_copy ), sizeof( rf_mp_real_t ) *
            ::NS(RFMultiPole_num_phase_elements)( rfmp_02 ) ) == 0 );

    /* --------------------------------------------------------------------- */

    rfmp_01 = ::NS(RFMultiPole_from_managed_buffer)(
        ::NS(Buffer_get_data_begin)( lattice ), rfmp_01_idx,
            ::NS(Buffer_get_slot_size)( lattice ) );

    ASSERT_TRUE( rfmp_01 != nullptr );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(RFMultiPole_copy)(
        rfmp_01, rfmp_02_copy) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(RFMultiPole_copy)(
        rfmp_02_copy, rfmp_01 ) );

    /* --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( lattice );
}
