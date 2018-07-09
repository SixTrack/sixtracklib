#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/impl/faddeeva.h"

TEST( CommonFaddeevaErrfnTests, CompareWCern355WithTabulatedValues )
{
    std::string const PATH_TO_TABULATED_DATA(
        st_PATH_TO_TEST_FADDEEVA_ERRFN_DATA );

    ::FILE* fp = std::fopen( PATH_TO_TABULATED_DATA.c_str(), "rb" );

    ASSERT_TRUE( fp != 0 );

    double const CMP_EPS = 9e-06;

    uint64_t N = 0;
    int cnt = std::fread( &N, sizeof( N ), 1u, fp );
    ASSERT_TRUE( cnt == 1u );

    double min_real = std::numeric_limits< double >::max();
    double min_imag = min_real;

    double max_real = -min_real;
    double max_imag = max_real;

    for( uint64_t ii = 0 ; ii < N ; ++ii )
    {
        double z_real          = double{ 0.0 };
        double z_imag          = double{ 0.0 };

        double cmp_result_real = double{ 0.0 };
        double cmp_result_imag = double{ 0.0 };

        cnt = std::fread( &z_real, sizeof( double ), 1u, fp );
        ASSERT_TRUE( cnt == 1u );

        cnt = std::fread( &z_imag, sizeof( double ), 1u, fp );
        ASSERT_TRUE( cnt == 1u );

        cnt = std::fread( &cmp_result_real, sizeof( double ), 1u, fp );
        ASSERT_TRUE( cnt == 1u );

        cnt = std::fread( &cmp_result_imag, sizeof( double ), 1u, fp );
        ASSERT_TRUE( cnt == 1u );

        if( z_real > max_real ) max_real = z_real;
        if( z_real < min_real ) min_real = z_real;

        if( z_imag > max_imag ) max_imag = z_imag;
        if( z_imag < min_imag ) min_imag = z_imag;

        double result_real = double{ 0.0 };
        double result_imag = double{ 0.0 };

        int ret = ::st_Faddeeva_calculate_w(
            &result_real, &result_imag, z_real, z_imag );

        ASSERT_TRUE( ret == 0 );

        double const diff_real  = std::fabs( cmp_result_real - result_real );
        double const diff_imag  = std::fabs( cmp_result_imag - result_imag );

        if( diff_real > CMP_EPS )
        {
            std::cout << "ii = " << ii
                      << " | diff_real = " << diff_real << std::endl;
        }

        if( diff_imag > CMP_EPS )
        {
            std::cout << "ii = " << ii
                      << " | diff_imag = " << diff_imag << std::endl;
        }

        bool const diff_real_valid = ( diff_real   < CMP_EPS );
        bool const diff_imag_valid = ( diff_imag   < CMP_EPS );

        ASSERT_TRUE( diff_real_valid );
        ASSERT_TRUE( diff_imag_valid );
    }

    std::cout << std::endl
              << "Successfully Tested in Range \r\n"
              << "  - real : min = " << std::setw( 8 ) << min_real << "\r\n"
              << "           max = " << std::setw( 8 ) << max_real << "\r\n"
              << "\r\n"
              << "  - imag : min = " << std::setw( 8 ) << min_imag << "\r\n"
              << "           max = " << std::setw( 8 ) << max_imag << "\r\n"
              << "\r\n"
              << "  - tolerance level for comparison with \r\n"
              << "    scipy.special.wofz: " << CMP_EPS
              << "\r\n"
              << std::endl;

    if( fp != nullptr )
    {
        std::fclose( fp );
        fp = nullptr;
    }
}

/* end: tests/sixtracklib/common/test_faddeeva_errf.cpp */
