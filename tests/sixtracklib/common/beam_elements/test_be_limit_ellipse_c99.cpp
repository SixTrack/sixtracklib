#include "sixtracklib/common/be_limit/be_limit_ellipse.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/constants.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_limit/track.h"

TEST( C99_CommonBeamElementLimitEllipseTests, BasicUsage )
{
    using be_limit_t   = ::NS(LimitEllipse);
    using buffer_t     = ::NS(Buffer);
    using size_t       = ::NS(buffer_size_t);
    using real_t       = ::NS(particle_real_t);
    
    using real_limit_t = std::numeric_limits< real_t >;
    real_t const EPS   = real_limit_t::epsilon();
    
    real_t const X_HALF_AXIS = real_t{ 2.0 };
    real_t const Y_HALF_AXIS = real_t{ 1.5 };
    
    be_limit_t limit;    
    ::NS(LimitEllipse_set_half_axes)( &limit, X_HALF_AXIS, Y_HALF_AXIS );
        
    buffer_t* eb = ::NS(Buffer_new)( size_t{ 0 } );
    
    be_limit_t* l2 = ::NS(LimitEllipse_new)( eb );    
    ASSERT_TRUE( l2 != nullptr );
            
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_x_half_axis)( l2 ) - 
        ::NS(LIMIT_DEFAULT_X_HALF_AXIS) ) < EPS );
    
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_y_half_axis)( l2 ) - 
        ::NS(LIMIT_DEFAULT_Y_HALF_AXIS) ) < EPS );
        
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_x_half_axis_squ)( l2 ) - 
        ::NS(LIMIT_DEFAULT_X_HALF_AXIS) * 
        ::NS(LIMIT_DEFAULT_X_HALF_AXIS) ) < EPS );
    
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_y_half_axis_squ)( l2 ) - 
        ::NS(LIMIT_DEFAULT_Y_HALF_AXIS) * 
        ::NS(LIMIT_DEFAULT_Y_HALF_AXIS) ) < EPS );
        
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_half_axes_product_squ)( l2 ) 
        - ::NS(LIMIT_DEFAULT_X_HALF_AXIS) * ::NS(LIMIT_DEFAULT_X_HALF_AXIS) * 
          ::NS(LIMIT_DEFAULT_Y_HALF_AXIS) * ::NS(LIMIT_DEFAULT_Y_HALF_AXIS) ) 
        < EPS );
    
    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == 
                 ::NS(LimitEllipse_copy)( &limit, l2 ) );
    
    ASSERT_TRUE( 0 == ::NS(LimitEllipse_compare_values)( &limit, l2 ) );
    
    be_limit_t* l3 = ::NS(LimitEllipse_add)( eb, X_HALF_AXIS, Y_HALF_AXIS );
    ASSERT_TRUE( l3 != nullptr );
    
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_x_half_axis)( l3 ) - 
        X_HALF_AXIS ) < EPS );
    
    ASSERT_TRUE( std::fabs( ::NS(LimitEllipse_get_y_half_axis)( l3 ) - 
        Y_HALF_AXIS ) < EPS );
    
    be_limit_t* l4 = ::NS(LimitEllipse_add_copy)( eb, &limit );
    
    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(LimitEllipse_compare_values)( l4, &limit ) );
    
    real_t const TRESHOLD = real_t{ 9e-4 };
    
    ::NS(LimitEllipse_set_half_axes_squ)( l4, 
        ::NS(LimitEllipse_get_x_half_axis_squ)( l4 ) + TRESHOLD, 
        ::NS(LimitEllipse_get_y_half_axis_squ)( l4 ) );
        
    ASSERT_TRUE( 0 != ::NS(LimitEllipse_compare_values)( l4, &limit ) );
    
    ASSERT_TRUE( 0 != ::NS(LimitEllipse_compare_values_with_treshold)( 
        l4, &limit, EPS ) );
    
    ASSERT_TRUE( 0 == ::NS(LimitEllipse_compare_values_with_treshold)( 
        l4, &limit, TRESHOLD ) );
    
    ::NS(Buffer_delete)( eb );
    eb = nullptr;
}    


TEST( C99_CommonBeamElementLimitRectTests, ApertureCheck )
{
    using be_limit_t  = ::NS(LimitEllipse);
    using buffer_t    = ::NS(Buffer);
    using particles_t = ::NS(Particles);
    using size_t      = ::NS(buffer_size_t);
    using real_t      = ::NS(particle_real_t);
    using pindex_t    = ::NS(particle_index_t);
    
    size_t const NUM_PARTICLES = size_t{ 20 };
    
    real_t const EPS = std::numeric_limits< real_t >::epsilon();
    pindex_t const LOST_PARTICLE = pindex_t{ 0 };
    pindex_t const NOT_LOST_PARTICLE = pindex_t{ 1 };
    
    std::vector< pindex_t > expected_state_after_track( 
        NUM_PARTICLES, LOST_PARTICLE );
    
    be_limit_t limit;
    
    real_t const ZERO = real_t{ 0.0 };
    
    real_t const X_HALF_AXIS = real_t{ +1.0 };
    real_t const Y_HALF_AXIS = real_t{ +1.0 };
    
    ::NS(LimitEllipse_set_half_axes)( &limit, X_HALF_AXIS, Y_HALF_AXIS );
    
    buffer_t* pb = ::NS(Buffer_new)( size_t{ 0 } );
    SIXTRL_ASSERT( pb != nullptr );
    
    particles_t* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );
    
    ::NS(Particles_set_x_value)(     particles,  0, -X_HALF_AXIS );
    ::NS(Particles_set_y_value)(     particles,  0, ZERO );
    ::NS(Particles_set_state_value)( particles,  0, NOT_LOST_PARTICLE );
    expected_state_after_track[  0 ] = NOT_LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  1, X_HALF_AXIS );
    ::NS(Particles_set_y_value)(     particles,  1, ZERO );
    ::NS(Particles_set_state_value)( particles,  1, NOT_LOST_PARTICLE );
    expected_state_after_track[  1 ] = NOT_LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  2, ZERO );
    ::NS(Particles_set_y_value)(     particles,  2, -Y_HALF_AXIS );
    ::NS(Particles_set_state_value)( particles,  2, NOT_LOST_PARTICLE );    
    expected_state_after_track[  2 ] = NOT_LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  3, ZERO );
    ::NS(Particles_set_y_value)(     particles,  3, Y_HALF_AXIS );
    ::NS(Particles_set_state_value)( particles,  3, NOT_LOST_PARTICLE );
    expected_state_after_track[  3 ] = NOT_LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  4, -X_HALF_AXIS - EPS );
    ::NS(Particles_set_y_value)(     particles,  4, ZERO );
    ::NS(Particles_set_state_value)( particles,  4, NOT_LOST_PARTICLE );
    expected_state_after_track[  4 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  5, X_HALF_AXIS + EPS );
    ::NS(Particles_set_y_value)(     particles,  5, ZERO );
    ::NS(Particles_set_state_value)( particles,  5, NOT_LOST_PARTICLE );
    expected_state_after_track[  5 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  6, ZERO );
    ::NS(Particles_set_y_value)(     particles,  6, -Y_HALF_AXIS - EPS );
    ::NS(Particles_set_state_value)( particles,  6, NOT_LOST_PARTICLE );
    expected_state_after_track[  6 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  7, ZERO );
    ::NS(Particles_set_y_value)(     particles,  7, Y_HALF_AXIS + EPS );
    ::NS(Particles_set_state_value)( particles,  7, NOT_LOST_PARTICLE );
    expected_state_after_track[  7 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  8, -X_HALF_AXIS );
    ::NS(Particles_set_y_value)(     particles,  8, ZERO );
    ::NS(Particles_set_state_value)( particles,  8, LOST_PARTICLE );
    expected_state_after_track[  8 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles,  9, X_HALF_AXIS );
    ::NS(Particles_set_y_value)(     particles,  9, ZERO );
    ::NS(Particles_set_state_value)( particles,  9, LOST_PARTICLE );
    expected_state_after_track[  9 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 10, ZERO );
    ::NS(Particles_set_y_value)(     particles, 10, -Y_HALF_AXIS );
    ::NS(Particles_set_state_value)( particles, 10, LOST_PARTICLE );    
    expected_state_after_track[ 10 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 11, ZERO );
    ::NS(Particles_set_y_value)(     particles, 11, Y_HALF_AXIS );
    ::NS(Particles_set_state_value)( particles, 11, LOST_PARTICLE );
    expected_state_after_track[ 11 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 12, -X_HALF_AXIS - EPS );
    ::NS(Particles_set_y_value)(     particles, 12, ZERO );
    ::NS(Particles_set_state_value)( particles, 12, LOST_PARTICLE );
    expected_state_after_track[ 12 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 13, X_HALF_AXIS + EPS );
    ::NS(Particles_set_y_value)(     particles, 13, ZERO );
    ::NS(Particles_set_state_value)( particles, 13, LOST_PARTICLE );
    expected_state_after_track[ 13 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 14, ZERO );
    ::NS(Particles_set_y_value)(     particles, 14, -Y_HALF_AXIS - EPS );
    ::NS(Particles_set_state_value)( particles, 14, LOST_PARTICLE );
    expected_state_after_track[ 14 ] = LOST_PARTICLE;
    
    ::NS(Particles_set_x_value)(     particles, 15, ZERO );
    ::NS(Particles_set_y_value)(     particles, 15, Y_HALF_AXIS + EPS );
    ::NS(Particles_set_state_value)( particles, 15, LOST_PARTICLE );
    expected_state_after_track[ 15 ] = LOST_PARTICLE;
    
    /* --------------------------------------------------------------------- */
    
    ::NS(Particles_set_x_value)( particles, 16, real_t{ 0.99 } * X_HALF_AXIS * 
        std::cos( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_y_value)( particles, 16, real_t{ 0.99 } * Y_HALF_AXIS * 
        std::sin( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_state_value)( particles, 16, NOT_LOST_PARTICLE );
    expected_state_after_track[ 16 ] = NOT_LOST_PARTICLE;
    
    
    
    ::NS(Particles_set_x_value)( particles, 17, real_t{ 1.02 } * X_HALF_AXIS * 
        std::cos( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_y_value)( particles, 17, real_t{ 1.02 } * Y_HALF_AXIS * 
        std::sin( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_state_value)( particles, 17, NOT_LOST_PARTICLE );
    expected_state_after_track[ 17 ] = LOST_PARTICLE;
    
    
    
    ::NS(Particles_set_x_value)( particles, 18, real_t{ 0.99 } * X_HALF_AXIS * 
        std::cos( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_y_value)( particles, 18, real_t{ 0.99 } * Y_HALF_AXIS * 
        std::sin( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_state_value)( particles, 18, LOST_PARTICLE );
    expected_state_after_track[ 18 ] = LOST_PARTICLE;
    
    
    
    ::NS(Particles_set_x_value)( particles, 19, real_t{ 1.02 } * X_HALF_AXIS * 
        std::cos( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_y_value)( particles, 19, real_t{ 1.02 } * Y_HALF_AXIS * 
        std::sin( ::NS(DEG2RAD) * real_t{ 30.0 } ) );
    
    ::NS(Particles_set_state_value)( particles, 19, LOST_PARTICLE );
    expected_state_after_track[ 19 ] = LOST_PARTICLE;
    
    /* --------------------------------------------------------------------- */
    
    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(TRACK_SUCCESS) == ::NS(Track_particle_limit_ellipse)(
            particles, ii, &limit ) );
        
        ASSERT_TRUE( ::NS(Particles_get_state_value)( particles, ii ) ==
                     expected_state_after_track[ ii ] );
    }
    
    ::NS(Buffer_delete)( pb );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_c99.cpp */
