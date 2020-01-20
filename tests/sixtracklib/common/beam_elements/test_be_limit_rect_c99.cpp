#include "sixtracklib/common/be_limit/be_limit_rect.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_limit/track.h"

TEST( C99CommonBeamElementLimitRectTests, BasicUsage )
{
    using be_limit_t   = ::NS(LimitRect);
    using buffer_t     = ::NS(Buffer);
    using size_t       = ::NS(buffer_size_t);
    using real_t       = ::NS(particle_real_t);

    using real_limit_t = std::numeric_limits< real_t >;
    real_t const EPS   = real_limit_t::epsilon();

    real_t const MIN_X_VALUE = real_t{ -2.0 };
    real_t const MAX_X_VALUE = real_t{  2.0 };

    real_t const MIN_Y_VALUE = real_t{ -3.0 };
    real_t const MAX_Y_VALUE = real_t{  3.0 };

    be_limit_t limit;
    ::NS(LimitRect_preset)( &limit );
    ::NS(LimitRect_set_min_x)( &limit, MIN_X_VALUE );
    ::NS(LimitRect_set_max_x)( &limit, MAX_X_VALUE );
    ::NS(LimitRect_set_min_y)( &limit, MIN_Y_VALUE );
    ::NS(LimitRect_set_max_y)( &limit, MAX_Y_VALUE );


    buffer_t* eb = ::NS(Buffer_new)( size_t{ 0 } );

    be_limit_t* l2 = ::NS(LimitRect_new)( eb );
    ASSERT_TRUE( l2 != nullptr );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_min_x)( l2 ) -
        ::NS(LIMIT_DEFAULT_MIN_X) ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_max_x)( l2 ) -
        ::NS(LIMIT_DEFAULT_MAX_X) ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_min_y)( l2 ) -
        ::NS(LIMIT_DEFAULT_MIN_Y) ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_max_y)( l2 ) -
        ::NS(LIMIT_DEFAULT_MAX_Y) ) < EPS );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
                 ::NS(LimitRect_copy)( &limit, l2 ) );
    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)( &limit, l2 ) );

    be_limit_t* l3 = ::NS(LimitRect_add)( eb,
        MIN_X_VALUE, MAX_X_VALUE, MIN_Y_VALUE, MAX_Y_VALUE );
    ASSERT_TRUE( l3 != nullptr );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_min_x)( l3 ) -
        MIN_X_VALUE ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_max_x)( l3 ) -
        MAX_X_VALUE ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_min_y)( l3 ) -
        MIN_Y_VALUE ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_max_y)( l3 ) -
        MAX_Y_VALUE ) < EPS );

    ::NS(LimitRect_set_x_limit)( l3, ::NS(LimitRect_get_max_y)( l3 ) );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_max_x)( l3 ) -
        ::NS(LimitRect_get_max_y)( l3 ) ) < EPS );

    ASSERT_TRUE( std::fabs( ::NS(LimitRect_get_min_x)( l3 ) -
        ( -::NS(LimitRect_get_max_y)( l3 ) ) ) < EPS );


    be_limit_t* l4 = ::NS(LimitRect_add_copy)( eb, &limit );

    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values)( l4, &limit ) );

    real_t const TRESHOLD = real_t{ 9e-4 };

    ::NS(LimitRect_set_x_limit)( l4,
         ::NS(LimitRect_get_max_x)( l4 ) + TRESHOLD );

    ASSERT_TRUE( 0 != ::NS(LimitRect_compare_values)( l4, &limit ) );

    ASSERT_TRUE( 0 != ::NS(LimitRect_compare_values_with_treshold)(
        l4, &limit, EPS ) );

    ASSERT_TRUE( 0 == ::NS(LimitRect_compare_values_with_treshold)(
        l4, &limit, TRESHOLD ) );

    ::NS(Buffer_delete)( eb );
}


TEST( C99CommonBeamElementLimitRectTests, ApertureCheck )
{
    using be_limit_t  = ::NS(LimitRect);
    using buffer_t    = ::NS(Buffer);
    using particles_t = ::NS(Particles);
    using size_t      = ::NS(buffer_size_t);
    using real_t      = ::NS(particle_real_t);
    using pindex_t    = ::NS(particle_index_t);

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

    ::NS(LimitRect_set_min_x)( &limit, LIMIT_MIN_X );
    ::NS(LimitRect_set_max_x)( &limit, LIMIT_MAX_X );
    ::NS(LimitRect_set_min_y)( &limit, LIMIT_MIN_Y );
    ::NS(LimitRect_set_max_y)( &limit, LIMIT_MAX_Y );

    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    SIXTRL_ASSERT( pb != nullptr );

    particles_t* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );

    ::NS(Particles_set_x_value)(     particles,  0, LIMIT_MIN_X );
    ::NS(Particles_set_y_value)(     particles,  0, ZERO );
    ::NS(Particles_set_state_value)( particles,  0, NOT_LOST_PARTICLE );
    expected_state_after_track[  0 ] = NOT_LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  1, LIMIT_MAX_X );
    ::NS(Particles_set_y_value)(     particles,  1, ZERO );
    ::NS(Particles_set_state_value)( particles,  1, NOT_LOST_PARTICLE );
    expected_state_after_track[  1 ] = NOT_LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  2, ZERO );
    ::NS(Particles_set_y_value)(     particles,  2, LIMIT_MIN_Y );
    ::NS(Particles_set_state_value)( particles,  2, NOT_LOST_PARTICLE );
    expected_state_after_track[  2 ] = NOT_LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  3, ZERO );
    ::NS(Particles_set_y_value)(     particles,  3, LIMIT_MAX_Y );
    ::NS(Particles_set_state_value)( particles,  3, NOT_LOST_PARTICLE );
    expected_state_after_track[  3 ] = NOT_LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  4, LIMIT_MIN_X - EPS );
    ::NS(Particles_set_y_value)(     particles,  4, ZERO );
    ::NS(Particles_set_state_value)( particles,  4, NOT_LOST_PARTICLE );
    expected_state_after_track[  4 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  5, LIMIT_MAX_X + EPS );
    ::NS(Particles_set_y_value)(     particles,  5, ZERO );
    ::NS(Particles_set_state_value)( particles,  5, NOT_LOST_PARTICLE );
    expected_state_after_track[  5 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  6, ZERO );
    ::NS(Particles_set_y_value)(     particles,  6, LIMIT_MIN_Y - EPS );
    ::NS(Particles_set_state_value)( particles,  6, NOT_LOST_PARTICLE );
    expected_state_after_track[  6 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  7, ZERO );
    ::NS(Particles_set_y_value)(     particles,  7, LIMIT_MAX_Y + EPS );
    ::NS(Particles_set_state_value)( particles,  7, NOT_LOST_PARTICLE );
    expected_state_after_track[  7 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  8, LIMIT_MIN_X );
    ::NS(Particles_set_y_value)(     particles,  8, ZERO );
    ::NS(Particles_set_state_value)( particles,  8, LOST_PARTICLE );
    expected_state_after_track[  8 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles,  9, LIMIT_MAX_X );
    ::NS(Particles_set_y_value)(     particles,  9, ZERO );
    ::NS(Particles_set_state_value)( particles,  9, LOST_PARTICLE );
    expected_state_after_track[  9 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 10, ZERO );
    ::NS(Particles_set_y_value)(     particles, 10, LIMIT_MIN_Y );
    ::NS(Particles_set_state_value)( particles, 10, LOST_PARTICLE );
    expected_state_after_track[ 10 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 11, ZERO );
    ::NS(Particles_set_y_value)(     particles, 11, LIMIT_MAX_Y );
    ::NS(Particles_set_state_value)( particles, 11, LOST_PARTICLE );
    expected_state_after_track[ 11 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 12, LIMIT_MIN_X - EPS );
    ::NS(Particles_set_y_value)(     particles, 12, ZERO );
    ::NS(Particles_set_state_value)( particles, 12, LOST_PARTICLE );
    expected_state_after_track[ 12 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 13, LIMIT_MAX_X + EPS );
    ::NS(Particles_set_y_value)(     particles, 13, ZERO );
    ::NS(Particles_set_state_value)( particles, 13, LOST_PARTICLE );
    expected_state_after_track[ 13 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 14, ZERO );
    ::NS(Particles_set_y_value)(     particles, 14, LIMIT_MIN_Y - EPS );
    ::NS(Particles_set_state_value)( particles, 14, LOST_PARTICLE );
    expected_state_after_track[ 14 ] = LOST_PARTICLE;

    ::NS(Particles_set_x_value)(     particles, 15, ZERO );
    ::NS(Particles_set_y_value)(     particles, 15, LIMIT_MAX_Y + EPS );
    ::NS(Particles_set_state_value)( particles, 15, LOST_PARTICLE );
    expected_state_after_track[ 15 ] = LOST_PARTICLE;

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(TRACK_SUCCESS) == ::NS(Track_particle_limit_rect)(
            particles, ii, &limit ) );

        ASSERT_TRUE( ::NS(Particles_get_state_value)( particles, ii ) ==
                     expected_state_after_track[ ii ] );
    }

    ::NS(Buffer_delete)( pb );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_c99.cpp */
