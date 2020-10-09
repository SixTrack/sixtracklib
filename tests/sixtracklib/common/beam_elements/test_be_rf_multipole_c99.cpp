#include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "sixtracklib/testlib.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

TEST( C99CommonBeamElementsRFMultipole, StoreAndRestoreTests )
{
    using rf_multipole_t = ::NS(RFMultipole);
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

    rf_multipole_t* rfmp_01 = ::NS(RFMultipole_new)( lattice, rfmp_01_order );

    ASSERT_TRUE( rfmp_01_idx < ::NS(Buffer_get_num_of_objects)( lattice ) );
    ASSERT_TRUE( rfmp_01 != nullptr );
    ASSERT_TRUE( ::NS(RFMultipole_order)( rfmp_01 ) == rfmp_01_order );

    ASSERT_TRUE( ::NS(RFMultipole_bal_length)( rfmp_01 ) ==
        static_cast< rf_mp_int_t >( 2 * rfmp_01_order + 2 ) );

    ASSERT_TRUE( ::NS(RFMultipole_phase_length)( rfmp_01 ) ==
        static_cast< rf_mp_int_t >( 2 * rfmp_01_order + 2 ) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_voltage)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_frequency)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_lag)( rfmp_01 ) -
        rf_mp_real_t{ 0.0 } ) < REAL_EPS );

    ASSERT_TRUE( ::NS(RFMultipole_const_bal_begin)( rfmp_01 ) != nullptr );
    ASSERT_TRUE( ::NS(RFMultipole_const_phase_begin)( rfmp_01 ) != nullptr );

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

    rf_multipole_t* rfmp_02 = ::NS(RFMultipole_add)( lattice, rfmp_02_order,
        rfmp_02_voltage, rfmp_02_frequency, rfmp_02_lag,
            reinterpret_cast< std::uintptr_t >( &bal_values[ 0 ] ),
                std::uintptr_t{ 0 } );

    ASSERT_TRUE( rfmp_02_idx < ::NS(Buffer_get_num_of_objects)( lattice ) );
    ASSERT_TRUE( rfmp_02 != nullptr );
    ASSERT_TRUE( ::NS(RFMultipole_order)( rfmp_02 ) == rfmp_02_order );

    ASSERT_TRUE( ::NS(RFMultipole_bal_length)( rfmp_02 ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultipole_phase_length)( rfmp_02 ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_voltage)(
        rfmp_02 ) - rfmp_02_voltage ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_frequency)(
        rfmp_02 ) - rfmp_02_frequency ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_lag)(
        rfmp_02 ) - rfmp_02_lag ) < REAL_EPS );

    ASSERT_TRUE( ::NS(RFMultipole_const_bal_begin)( rfmp_02 ) != nullptr );
    ASSERT_TRUE( ::NS(RFMultipole_const_phase_begin)( rfmp_02 ) != nullptr );

    ASSERT_TRUE( 0 == std::memcmp( &bal_values[ 0 ],
        ::NS(RFMultipole_const_bal_begin)( rfmp_02 ),
        sizeof( rf_mp_real_t ) * ::NS(RFMultipole_bal_length)( rfmp_02 ) ) );

    /* --------------------------------------------------------------------- */

    buf_size_t const rfmp_02_copy_idx =
        ::NS(Buffer_get_num_of_objects)( lattice );

    rf_multipole_t* rfmp_02_copy =
        ::NS(RFMultipole_new)( lattice, rfmp_02_order );

    rfmp_02 = ::NS(RFMultipole_from_buffer)( lattice, rfmp_02_idx );

    ASSERT_TRUE( rfmp_02_copy_idx > rfmp_02_idx );
    ASSERT_TRUE( rfmp_02_copy_idx <
                    ::NS(Buffer_get_num_of_objects)( lattice ) );

    ASSERT_TRUE( rfmp_02 != nullptr );
    ASSERT_TRUE( rfmp_02_copy != nullptr );
    ASSERT_TRUE( rfmp_02_copy != rfmp_02 );
    ASSERT_TRUE( ::NS(RFMultipole_order)( rfmp_02 ) ==
                 ::NS(RFMultipole_order)( rfmp_02_copy ) );

    ASSERT_TRUE( ::NS(RFMultipole_bal_length)( rfmp_02_copy ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultipole_phase_length)( rfmp_02_copy ) ==
        static_cast< rf_mp_int_t >( 2u * rfmp_02_order + 2u ) );

    ASSERT_TRUE( ::NS(RFMultipole_const_bal_begin)(
        rfmp_02_copy ) != nullptr );

    ASSERT_TRUE( ::NS(RFMultipole_const_phase_begin)(
        rfmp_02_copy ) != nullptr );

    ASSERT_TRUE( ::NS(RFMultipole_copy)( rfmp_02_copy, rfmp_02 ) ==
                 ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_voltage)(
        rfmp_02_copy ) - rfmp_02_voltage ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_frequency)(
        rfmp_02_copy ) - rfmp_02_frequency ) < REAL_EPS );

    ASSERT_TRUE( std::fabs( ::NS(RFMultipole_lag)(
        rfmp_02_copy ) - rfmp_02_lag ) < REAL_EPS );

    ASSERT_TRUE( std::memcmp( &bal_values[ 0 ],
        ::NS(RFMultipole_const_bal_begin)( rfmp_02_copy ),
            sizeof( rf_mp_real_t ) * ::NS(RFMultipole_bal_length)(
                rfmp_02_copy ) ) == 0 );

    ASSERT_TRUE( std::memcmp( ::NS(RFMultipole_const_bal_begin)( rfmp_02 ),
        ::NS(RFMultipole_const_bal_begin)( rfmp_02_copy ),
            sizeof( rf_mp_real_t ) * ::NS(RFMultipole_bal_length)(
                rfmp_02 ) ) == 0 );

    ASSERT_TRUE( std::memcmp( ::NS(RFMultipole_const_phase_begin)( rfmp_02 ),
        ::NS(RFMultipole_const_phase_begin)( rfmp_02_copy ),
            sizeof( rf_mp_real_t ) * ::NS(RFMultipole_phase_length)(
                rfmp_02 ) ) == 0 );

    /* --------------------------------------------------------------------- */

    rfmp_01 = ::NS(RFMultipole_from_managed_buffer)(
        ::NS(Buffer_get_data_begin)( lattice ), rfmp_01_idx,
            ::NS(Buffer_get_slot_size)( lattice ) );

    ASSERT_TRUE( rfmp_01 != nullptr );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(RFMultipole_copy)(
        rfmp_01, rfmp_02_copy) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(RFMultipole_copy)(
        rfmp_02_copy, rfmp_01 ) );

    /* --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( lattice );
}
