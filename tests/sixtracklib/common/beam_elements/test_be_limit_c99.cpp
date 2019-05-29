#include "sixtracklib/common/be_limit/be_limit.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

TEST( C99_CommonBeamElementLimitTests, BasicUsage )
{
    using be_limit_t   = ::NS(Limit);
    using buffer_t     = ::NS(Buffer);
    using size_t       = ::NS(buffer_size_t);
    using real_t       = ::NS(particle_real_t);
    
    using real_limit_t = std::numeric_limits< real_t >;
    
    be_limit_t limit;
    ::NS(Limit_preset)( &limit );
    
    ::NS(Limit_set_x_limit)( &limit, real_t{ 0.2 } );
    ::NS(Limit_set_y_limit)( &limit, real_t{ 0.3 } );
    
    buffer_t* eb = ::NS(Buffer_new)( size_t{ 0 } );
    
    be_limit_t* l2 = ::NS(Limit_new)( eb );
    
    ASSERT_TRUE( l2 != nullptr );
    
    ASSERT_TRUE( std::fabs( ::NS(Limit_get_x_limit)( l2 ) - 
        ::NS(DEFAULT_X_LIMIT) ) < real_limit_t::epsilon() );
    
    ASSERT_TRUE( std::fabs( ::NS(Limit_get_y_limit)( l2 ) - 
        ::NS(DEFAULT_Y_LIMIT) ) < real_limit_t::epsilon() );
    
    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(Limit_copy)( &limit, l2 ) );
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values)( &limit, l2 ) );
    
    be_limit_t* l3 = ::NS(Limit_add)( eb, 2.0, 3.0 );
    
    ASSERT_TRUE( l3 != nullptr );
    ASSERT_TRUE( std::fabs( 2.0 - ::NS(Limit_get_x_limit)( l3 ) ) < 
        real_limit_t::epsilon() );
    
    ASSERT_TRUE( std::fabs( 3.0 - ::NS(Limit_get_y_limit)( l3 ) ) <
        real_limit_t::epsilon() );
    
    ::NS(Limit_set_x_limit)( l3, ::NS(Limit_get_y_limit)( l3 ) );
    
    ASSERT_TRUE( std::fabs( ::NS(Limit_get_x_limit)( l3 ) - 
        ::NS(Limit_get_y_limit)( l3 ) ) < real_limit_t::epsilon() );
    
    be_limit_t* l4 = ::NS(Limit_add_copy)( eb, &limit );
    
    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values)( l4, &limit ) );
    
    ::NS(Limit_set_x_limit)( l4, 
         ::NS(Limit_get_x_limit)( l4 ) + real_t{ 9e-4 } );
    
    ASSERT_TRUE( 0 != ::NS(Limit_compare_values)( l4, &limit ) );
    ASSERT_TRUE( 0 != ::NS(Limit_compare_values_with_treshold)( l4, &limit, 
        real_limit_t::epsilon() ) );
    
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values_with_treshold)( l4, &limit, 
        real_t{ 9e-4 } ) );
    
    ::NS(Buffer_delete)( eb );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_c99.cpp */
