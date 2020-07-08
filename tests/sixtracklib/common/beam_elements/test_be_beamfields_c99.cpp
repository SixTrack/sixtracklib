#include "sixtracklib/common/be_beamfields/be_beamfields.h"
#include "sixtracklib/common/be_beamfields/track.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace tests
    {
        template< class T >
        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN T gaussian_dist(
            T const& x, T const& sigma, T const& mu = T{ 0 } )
        {
            namespace st = SIXTRL_CXX_NAMESPACE;
            SIXTRL_ASSERT( sigma > T{ 0 } );
            T const scaled_x = ( x - mu ) / sigma;
            return T{ 1 } / ( sigma * st::sqrt< T >( 2 ) *
                st::MathConst_sqrt_pi< T >() ) *
                st::exp< T >( - T{ 0.5 } * scaled_x * scaled_x );
        }
    }
}

TEST( C99CommonBeamElementBeamField, QGaussianDist )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using real_t = SIXTRL_REAL_T;

    ASSERT_TRUE( ::NS(Math_q_gaussian_cq)( real_t{ 3.0001 } ) == real_t{ 0 } );
    ASSERT_TRUE( ::NS(Math_q_gaussian_cq)( real_t{ 3 } ) == real_t{ 0 } );

    real_t const q0 = ( real_t )1.0;
    real_t const sigma_0 = 2.5;
    real_t const Cq_0 = ::NS(Math_q_gaussian_cq)( q0 );
    static real_t const ABS_TOL = std::numeric_limits< real_t >::max();

    ASSERT_TRUE( Cq_0 > real_t{ 0 } );
    ASSERT_TRUE( st::abs< real_t >( st::MathConst_sqrt_pi<
        real_t >() - Cq_0 ) <= ABS_TOL );

    real_t const sqrt_beta_0 =
        ::NS(Math_q_gaussian_sqrt_beta_from_gaussian_sigma)( sigma_0 );

    ASSERT_TRUE( sqrt_beta_0 > real_t{ 0 } );
    ASSERT_TRUE( st::Type_comp_all_are_close< real_t >(
        sqrt_beta_0 * sqrt_beta_0, real_t{ 1 } / (
            real_t{ 2 } * sigma_0 * sigma_0 ), real_t{ 0 }, ABS_TOL ) );

    real_t const xmin_0 = -real_t{ 6 } * sigma_0;
    real_t const xmax_0 = +real_t{ 6 } * sigma_0;
    real_t const dx_0 = ( xmax_0 - xmin_0 ) / real_t{ 1000 };

    real_t x = xmin_0;

    while( x < xmax_0 )
    {
        real_t const gaussian_x = st::tests::gaussian_dist( x, sigma_0 );
        real_t const q_gaussian_x = ::NS(Math_q_gaussian)(
            x, q0, sqrt_beta_0, Cq_0 );

        ASSERT_TRUE( st::abs< real_t >( gaussian_x - q_gaussian_x ) < ABS_TOL );
        x += dx_0;
    }

    /* q < 1 */

    real_t const q1 = real_t{ 3 } / real_t{ 5 };
    real_t const Cq_1 = ::NS(Math_q_gaussian_cq)( q1 );
    ASSERT_TRUE( Cq_1 > real_t{ 0 } );

    real_t const sqrt_beta_1 = st::sqrt< real_t >( real_t{ 2 } );
    real_t const xmin_1_support = real_t{ -1 } / st::sqrt< real_t >(
        sqrt_beta_1 * sqrt_beta_1 * ( real_t{ 1 } - q1 ) );

    real_t const xmax_1_support = real_t{ +1 } / st::sqrt< real_t >(
        sqrt_beta_1 * sqrt_beta_1 * ( real_t{ 1 } - q1 ) );

    real_t const xmin_1 = real_t{ 2 } * xmin_1_support;
    real_t const xmax_1 = real_t{ 2 } * xmax_1_support;
    real_t const dx_1 = ( xmax_1_support - xmin_1_support ) / real_t{ 100 };

    x = xmin_1;

    while( x < xmax_1 )
    {
        real_t const q_gauss_x = ::NS(Math_q_gaussian)(
            x, q1, sqrt_beta_1, Cq_1 );

        std::cout << std::setw( 20 ) << x << std::setw( 20 )
                  << q_gauss_x << "\r\n";

        if( ( x < xmin_1_support ) || ( x > xmax_1_support ) )
        {
            ASSERT_TRUE( st::abs< real_t >( q_gauss_x ) <= ABS_TOL );
        }
        else
        {
            ASSERT_TRUE( q_gauss_x > real_t{ 0 } );
        }

        x += dx_1;
    }

    std::cout << std::endl;

}
