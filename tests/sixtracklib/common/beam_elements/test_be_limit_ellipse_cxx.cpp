#include "sixtracklib/common/be_limit/be_limit_ellipse.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/be_limit/track.h"

TEST( CXXCommonBeamElementLimitEllipse, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using be_limit_t   = st::LimitEllipse;
    using buffer_t     = st::Buffer;
    using real_t       = st::Particles::real_t;

    real_t const EPS   = std::numeric_limits< real_t >::epsilon();

    real_t const X_HALF_AXIS = real_t{ 2.0 };
    real_t const Y_HALF_AXIS = real_t{ 1.5 };

    be_limit_t limit;
    limit.setHalfAxes( X_HALF_AXIS, Y_HALF_AXIS );

    buffer_t eb;

    be_limit_t* l2 = eb.createNew< be_limit_t >();
    ASSERT_TRUE( l2 != nullptr );

    ASSERT_TRUE( std::fabs( l2->getXHalfAxis() -
        st::LIMIT_DEFAULT_X_HALF_AXIS ) < EPS );

    ASSERT_TRUE( std::fabs( l2->getYHalfAxis() -
        st::LIMIT_DEFAULT_Y_HALF_AXIS ) < EPS );

    ASSERT_TRUE( std::fabs( l2->getXHalfAxisSqu() -
        st::LIMIT_DEFAULT_X_HALF_AXIS * st::LIMIT_DEFAULT_X_HALF_AXIS ) < EPS );

    ASSERT_TRUE( std::fabs( l2->getYHalfAxisSqu() -
        st::LIMIT_DEFAULT_Y_HALF_AXIS * st::LIMIT_DEFAULT_Y_HALF_AXIS ) < EPS );

    ASSERT_TRUE( std::fabs( l2->getHalfAxesProductSqu() -
        st::LIMIT_DEFAULT_X_HALF_AXIS * st::LIMIT_DEFAULT_X_HALF_AXIS *
        st::LIMIT_DEFAULT_Y_HALF_AXIS * st::LIMIT_DEFAULT_Y_HALF_AXIS ) < EPS );


    be_limit_t* l4 = eb.addCopy< be_limit_t >( limit );

    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(LimitEllipse_compare_values)(
        l4->getCApiPtr(), limit.getCApiPtr() ) );

    real_t const TRESHOLD = real_t{ 9e-4 };

    l4->setHalfAxesSqu( l4->getXHalfAxisSqu() + TRESHOLD,
                        l4->getYHalfAxisSqu() );

    ASSERT_TRUE( 0 != ::NS(LimitEllipse_compare_values)(
        l4->getCApiPtr(), limit.getCApiPtr() ) );

    ASSERT_TRUE( 0 != ::NS(LimitEllipse_compare_values_with_treshold)(
        l4->getCApiPtr(), limit.getCApiPtr(), EPS ) );
}

TEST( CXXCommonBeamElementLimitRect, ApertureCheck )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using be_limit_t  = st::LimitEllipse;
    using buffer_t    = st::Buffer;
    using particles_t = st::Particles;
    using size_t      = buffer_t::size_type;
    using real_t      = particles_t::real_t;
    using pindex_t    = particles_t::index_t;

    size_t const NUM_PARTICLES = size_t{ 20 };

    real_t const EPS = std::numeric_limits< real_t >::epsilon();
    pindex_t const LOST_PARTICLE = pindex_t{ 0 };
    pindex_t const NOT_LOST_PARTICLE = pindex_t{ 1 };

    std::vector< pindex_t > expected_state_after_track(
        NUM_PARTICLES, LOST_PARTICLE );

    be_limit_t limit;

    real_t const ZERO = real_t{ 0.0 };

    real_t const X_HALF_AXIS = real_t{ 1.0 };
    real_t const Y_HALF_AXIS = real_t{ 1.0 };

    limit.setHalfAxes( X_HALF_AXIS, Y_HALF_AXIS );

    buffer_t pb;

    particles_t* particles = pb.createNew< particles_t >( NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );

    particles->setXValue( 0, -X_HALF_AXIS );
    particles->setYValue( 0, ZERO );
    particles->setStateValue( 0, NOT_LOST_PARTICLE );
    expected_state_after_track[  0 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 1, X_HALF_AXIS );
    particles->setYValue( 1, ZERO );
    particles->setStateValue( 1, NOT_LOST_PARTICLE );
    expected_state_after_track[  1 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 2, ZERO );
    particles->setYValue( 2, -Y_HALF_AXIS );
    particles->setStateValue( 2, NOT_LOST_PARTICLE );
    expected_state_after_track[  2 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 3, ZERO );
    particles->setYValue( 3, Y_HALF_AXIS );
    particles->setStateValue( 3, NOT_LOST_PARTICLE );
    expected_state_after_track[  3 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 4, -X_HALF_AXIS - EPS );
    particles->setYValue( 4, ZERO );
    particles->setStateValue( 4, NOT_LOST_PARTICLE );
    expected_state_after_track[  4 ] = LOST_PARTICLE;

    particles->setXValue( 5, X_HALF_AXIS + EPS );
    particles->setYValue( 5, ZERO );
    particles->setStateValue( 5, NOT_LOST_PARTICLE );
    expected_state_after_track[  5 ] = LOST_PARTICLE;

    particles->setXValue( 6, ZERO );
    particles->setYValue( 6, -Y_HALF_AXIS - EPS );
    particles->setStateValue( 6, NOT_LOST_PARTICLE );
    expected_state_after_track[  6 ] = LOST_PARTICLE;

    particles->setXValue( 7, ZERO );
    particles->setYValue( 7, Y_HALF_AXIS + EPS );
    particles->setStateValue( 7, NOT_LOST_PARTICLE );
    expected_state_after_track[  7 ] = LOST_PARTICLE;

    particles->setXValue( 8, -X_HALF_AXIS );
    particles->setYValue( 8, ZERO );
    particles->setStateValue( 8, LOST_PARTICLE );
    expected_state_after_track[  8 ] = LOST_PARTICLE;

    particles->setXValue( 9, X_HALF_AXIS );
    particles->setYValue( 9, ZERO );
    particles->setStateValue( 9, LOST_PARTICLE );
    expected_state_after_track[  9 ] = LOST_PARTICLE;

    particles->setXValue( 10, ZERO );
    particles->setYValue( 10, -Y_HALF_AXIS );
    particles->setStateValue( 10, LOST_PARTICLE );
    expected_state_after_track[ 10 ] = LOST_PARTICLE;

    particles->setXValue( 11, ZERO );
    particles->setYValue( 11, Y_HALF_AXIS );
    particles->setStateValue( 11, LOST_PARTICLE );
    expected_state_after_track[ 11 ] = LOST_PARTICLE;

    particles->setXValue( 12, -X_HALF_AXIS - EPS );
    particles->setYValue( 12, ZERO );
    particles->setStateValue( 12, LOST_PARTICLE );
    expected_state_after_track[ 12 ] = LOST_PARTICLE;

    particles->setXValue( 13, X_HALF_AXIS + EPS );
    particles->setYValue( 13, ZERO );
    particles->setStateValue( 13, LOST_PARTICLE );
    expected_state_after_track[ 13 ] = LOST_PARTICLE;

    particles->setXValue( 14, ZERO );
    particles->setYValue( 14, -Y_HALF_AXIS - EPS );
    particles->setStateValue( 14, LOST_PARTICLE );
    expected_state_after_track[ 14 ] = LOST_PARTICLE;

    particles->setXValue( 15, ZERO );
    particles->setYValue( 15, Y_HALF_AXIS + EPS );
    particles->setStateValue( 15, LOST_PARTICLE );
    expected_state_after_track[ 15 ] = LOST_PARTICLE;

    /* --------------------------------------------------------------------- */

    particles->setXValue( 16, real_t{ 0.99 } * X_HALF_AXIS * std::cos(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setYValue( 16, real_t{ 0.99 } * Y_HALF_AXIS * std::sin(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setStateValue( 16, NOT_LOST_PARTICLE );
    expected_state_after_track[ 16 ] = NOT_LOST_PARTICLE;



    particles->setXValue( 17, real_t{ 1.02 } * X_HALF_AXIS * std::cos(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setYValue( 17, real_t{ 1.02 } * Y_HALF_AXIS * std::sin(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setStateValue( 17, NOT_LOST_PARTICLE );
    expected_state_after_track[ 17 ] = LOST_PARTICLE;



    particles->setXValue( 18, real_t{ 0.99 } * X_HALF_AXIS * std::cos(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setYValue( 18, real_t{ 0.99 } * Y_HALF_AXIS * std::sin(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setStateValue( 18, LOST_PARTICLE );
    expected_state_after_track[ 18 ] = LOST_PARTICLE;



    particles->setXValue( 19, real_t{ 1.02 } * X_HALF_AXIS * std::cos(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setYValue( 19, real_t{ 1.02 } * Y_HALF_AXIS * std::sin(
        ::NS(DEG2RAD) * real_t{ 30.0 } ) );

    particles->setStateValue( 19, LOST_PARTICLE );
    expected_state_after_track[ 19 ] = LOST_PARTICLE;

    /* --------------------------------------------------------------------- */

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(TRACK_SUCCESS) == ::NS(Track_particle_limit_ellipse)(
            particles->getCApiPtr(), ii, limit.getCApiPtr() ) );

        ASSERT_TRUE( particles->getStateValue( ii ) ==
                     expected_state_after_track[ ii ] );
    }
}
