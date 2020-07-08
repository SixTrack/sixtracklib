#include "sixtracklib/common/internal/math_interpol.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/internal/math_constants.h"
#include "sixtracklib/testlib.h"

TEST( C99MathInterpolLinear, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    std::size_t const NUM_PARTICLES = std::size_t{ 10 };
    std::vector< double > phi_abscissa(  NUM_PARTICLES, double{ 0 } );
    std::vector< double > cos_phi( NUM_PARTICLES, double{ 0 } );

    double const phi0  = -( st::MathConst_pi< double >() );
    double const d_phi = ( st::MathConst_pi< double >() - phi0 ) /
        static_cast< double >( NUM_PARTICLES - std::size_t{ 1 } );

    for( std::size_t ii = std::size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        phi_abscissa[ ii ] = phi0 + d_phi * ii;
        cos_phi[ ii ] = std::cos( phi_abscissa[ ii ] );
    }

    std::vector< double > cos_phi_derivative( NUM_PARTICLES, double{ 0 } );

    ASSERT_TRUE( NS(Math_interpol_prepare_equ)( cos_phi_derivative.data(),
        nullptr, cos_phi.data(), phi0, d_phi, phi_abscissa.size(),
            NS(MATH_INTERPOL_LINEAR) ) == st::ARCH_STATUS_SUCCESS );

    std::vector< double > temp_values( 6 * NUM_PARTICLES, double{ 0 } );

    ASSERT_TRUE( NS(Math_interpol_prepare_equ)( cos_phi_derivative.data(),
        temp_values.data(), cos_phi.data(), phi0, d_phi, phi_abscissa.size(),
            NS(MATH_INTERPOL_CUBIC) ) == st::ARCH_STATUS_SUCCESS );

    double phi = phi0;
    double const d = double{ 1e-4 };

    while( st::MathConst_pi< double >() > phi )
    {
        std::cout << std::setw( 20 ) << phi
                  << std::setw( 20 )
                  << ::NS(Math_interpol_y_equ)( phi, phi0, d_phi, cos_phi.data(),
                        cos_phi_derivative.data(), cos_phi.size(),
                        ::NS(MATH_INTERPOL_CUBIC) )
                  << std::setw( 20 )
                  << ::NS(Math_interpol_yp_equ)( phi, phi0, d_phi, cos_phi.data(),
                        cos_phi_derivative.data(), cos_phi.size(),
                        ::NS(MATH_INTERPOL_CUBIC) )
                  << std::setw( 20 )
                  << ::NS(Math_interpol_ypp_equ)( phi, phi0, d_phi, cos_phi.data(),
                        cos_phi_derivative.data(), cos_phi.size(),
                        ::NS(MATH_INTERPOL_CUBIC) )
                  << "\r\n";

        phi += d;
    }
}

/* end: tests/sixtracklib/common/control/test_node_id_c99.cpp */
