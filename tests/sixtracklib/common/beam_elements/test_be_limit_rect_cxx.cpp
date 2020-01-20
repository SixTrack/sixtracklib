#include "sixtracklib/common/be_limit/be_limit_rect.hpp"

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

TEST( C99CommonBeamElementLimitRectTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using be_limit_t   = st::LimitRect;
    using buffer_t     = st::Buffer;
    using real_t       = st::Particles::real_t;

    real_t const EPS   = std::numeric_limits< real_t >::epsilon();

    be_limit_t limit;
    limit.setXLimit( real_t{ 0.2 } );
    limit.setYLimit( real_t{ 0.3 } );

    buffer_t eb;

    be_limit_t* l2 = eb.createNew< be_limit_t >();
    ASSERT_TRUE( l2 != nullptr );

    ASSERT_TRUE( std::fabs( l2->getMinX() - be_limit_t::DEFAULT_MIN_X ) < EPS );
    ASSERT_TRUE( std::fabs( l2->getMaxX() - be_limit_t::DEFAULT_MAX_X ) < EPS );
    ASSERT_TRUE( std::fabs( l2->getMinY() - be_limit_t::DEFAULT_MIN_Y ) < EPS );
    ASSERT_TRUE( std::fabs( l2->getMaxY() - be_limit_t::DEFAULT_MAX_Y ) < EPS );

    ASSERT_TRUE( st::ARCH_STATUS_SUCCESS == ::NS(LimitRect_copy)(
        limit.getCApiPtr(), l2->getCApiPtr() ) );

    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)(
        limit.getCApiPtr(), l2->getCApiPtr() ) );


    real_t const MIN_X_VALUE = real_t{ -2.0 };
    real_t const MAX_X_VALUE = real_t{  2.0 };

    real_t const MIN_Y_VALUE = real_t{ -3.0 };
    real_t const MAX_Y_VALUE = real_t{  3.0 };

    be_limit_t* l3 = eb.add< be_limit_t >(
        MIN_X_VALUE, MAX_X_VALUE, MIN_Y_VALUE, MAX_Y_VALUE );

    ASSERT_TRUE( l3 != nullptr );

    ASSERT_TRUE( std::fabs( MIN_X_VALUE - l3->getMinX() ) < EPS );
    ASSERT_TRUE( std::fabs( MAX_X_VALUE - l3->getMaxX() ) < EPS );
    ASSERT_TRUE( std::fabs( MIN_Y_VALUE - l3->getMinY() ) < EPS );
    ASSERT_TRUE( std::fabs( MAX_Y_VALUE - l3->getMaxY() ) < EPS );


    be_limit_t* l4 = eb.addCopy< be_limit_t >( limit );

    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)(
        l4->getCApiPtr(), limit.getCApiPtr() ) );

    real_t const TRESHOLD = real_t{ 9e-4 };

    l4->setXLimit( l4->getMaxX() + TRESHOLD );

    ASSERT_TRUE( 0 != ::NS(LimitRect_compare_values)(
        l4->getCApiPtr(), limit.getCApiPtr() ) );

    ASSERT_TRUE( 0 != ::NS(LimitRect_compare_values_with_treshold)(
        l4->getCApiPtr(), limit.getCApiPtr(), EPS ) );

    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values_with_treshold)(
        l4->getCApiPtr(), limit.getCApiPtr(), TRESHOLD ) );
}


TEST( CXXCommonBeamElementLimitRectTests, ApertureCheck )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using be_limit_t  = st::LimitRect;
    using buffer_t    = st::Buffer;
    using particles_t = st::Particles;
    using size_t      = buffer_t::size_type;
    using real_t      = particles_t::real_t;
    using pindex_t    = particles_t::index_t;

    size_t const NUM_PARTICLES = size_t{ 16 };

    real_t const EPS = std::numeric_limits< real_t >::epsilon();
    pindex_t const LOST_PARTICLE = pindex_t{ 0 };
    pindex_t const NOT_LOST_PARTICLE = pindex_t{ 1 };

    std::vector< pindex_t > expected_state_after_track(
        NUM_PARTICLES, LOST_PARTICLE );

    be_limit_t limit;

    real_t const ZERO = real_t{ 0.0 };

    real_t const LIMIT_MIN_X = real_t{ -1.0 };
    real_t const LIMIT_MAX_X = real_t{ +1.0 };

    real_t const LIMIT_MIN_Y = real_t{ -1.0 };
    real_t const LIMIT_MAX_Y = real_t{ +1.0 };

    limit.setMinX( LIMIT_MIN_X );
    limit.setMaxX( LIMIT_MAX_X );
    limit.setMinY( LIMIT_MIN_Y );
    limit.setMaxY( LIMIT_MAX_Y );

    buffer_t pb;

    particles_t* particles = pb.createNew< particles_t >( NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );

    particles->setXValue( 0, LIMIT_MIN_X );
    particles->setYValue( 0, ZERO );
    particles->setStateValue( 0, NOT_LOST_PARTICLE );
    expected_state_after_track[  0 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 1, LIMIT_MAX_X );
    particles->setYValue( 1, ZERO );
    particles->setStateValue( 1, NOT_LOST_PARTICLE );
    expected_state_after_track[  1 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 2, ZERO );
    particles->setYValue( 2, LIMIT_MIN_Y );
    particles->setStateValue( 2, NOT_LOST_PARTICLE );
    expected_state_after_track[  2 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 3, ZERO );
    particles->setYValue( 3, LIMIT_MAX_Y );
    particles->setStateValue( 3, NOT_LOST_PARTICLE );
    expected_state_after_track[  3 ] = NOT_LOST_PARTICLE;

    particles->setXValue( 4, LIMIT_MIN_X - EPS );
    particles->setYValue( 4, ZERO );
    particles->setStateValue( 4, NOT_LOST_PARTICLE );
    expected_state_after_track[  4 ] = LOST_PARTICLE;

    particles->setXValue( 5, LIMIT_MAX_X + EPS );
    particles->setYValue( 5, ZERO );
    particles->setStateValue( 5, NOT_LOST_PARTICLE );
    expected_state_after_track[  5 ] = LOST_PARTICLE;

    particles->setXValue( 6, ZERO );
    particles->setYValue( 6, LIMIT_MIN_Y - EPS );
    particles->setStateValue( 6, NOT_LOST_PARTICLE );
    expected_state_after_track[  6 ] = LOST_PARTICLE;

    particles->setXValue( 7, ZERO );
    particles->setYValue( 7, LIMIT_MAX_Y + EPS );
    particles->setStateValue( 7, NOT_LOST_PARTICLE );
    expected_state_after_track[  7 ] = LOST_PARTICLE;

    particles->setXValue( 8, LIMIT_MIN_X );
    particles->setYValue( 8, ZERO );
    particles->setStateValue( 8, LOST_PARTICLE );
    expected_state_after_track[  8 ] = LOST_PARTICLE;

    particles->setXValue( 9, LIMIT_MAX_X );
    particles->setYValue( 9, ZERO );
    particles->setStateValue( 9, LOST_PARTICLE );
    expected_state_after_track[  9 ] = LOST_PARTICLE;

    particles->setXValue( 10, ZERO );
    particles->setYValue( 10, LIMIT_MIN_Y );
    particles->setStateValue( 10, LOST_PARTICLE );
    expected_state_after_track[ 10 ] = LOST_PARTICLE;

    particles->setXValue( 11, ZERO );
    particles->setYValue( 11, LIMIT_MAX_Y );
    particles->setStateValue( 11, LOST_PARTICLE );
    expected_state_after_track[ 11 ] = LOST_PARTICLE;

    particles->setXValue( 12, LIMIT_MIN_X - EPS );
    particles->setYValue( 12, ZERO );
    particles->setStateValue( 12, LOST_PARTICLE );
    expected_state_after_track[ 12 ] = LOST_PARTICLE;

    particles->setXValue( 13, LIMIT_MAX_X + EPS );
    particles->setYValue( 13, ZERO );
    particles->setStateValue( 13, LOST_PARTICLE );
    expected_state_after_track[ 13 ] = LOST_PARTICLE;

    particles->setXValue( 14, ZERO );
    particles->setYValue( 14, LIMIT_MIN_Y - EPS );
    particles->setStateValue( 14, LOST_PARTICLE );
    expected_state_after_track[ 14 ] = LOST_PARTICLE;

    particles->setXValue( 15, ZERO );
    particles->setYValue( 15, LIMIT_MAX_Y + EPS );
    particles->setStateValue( 15, LOST_PARTICLE );
    expected_state_after_track[ 15 ] = LOST_PARTICLE;

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(TRACK_SUCCESS) == ::NS(Track_particle_limit_rect)(
            particles->getCApiPtr(), ii, limit.getCApiPtr() ) );

        ASSERT_TRUE( particles->getStateValue( ii ) ==
                     expected_state_after_track[ ii ] );
    }
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_rect_cxx.cpp */
