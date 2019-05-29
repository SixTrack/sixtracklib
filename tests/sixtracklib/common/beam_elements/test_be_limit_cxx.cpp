#include "sixtracklib/common/be_limit/be_limit.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"

TEST( C99_CommonBeamElementLimitTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    
    using be_limit_t   = st::Limit;
    using buffer_t     = st::Buffer;
    using real_t       = st::Particles::real_t;    
    using real_limit_t = std::numeric_limits< real_t >;
    
    be_limit_t limit;
    limit.setXLimit( real_t{ 0.2 } );
    limit.setYLimit( real_t{ 0.3 } );
        
    buffer_t eb;
    
    be_limit_t* l2 = eb.createNew< be_limit_t >();
    ASSERT_TRUE( l2 != nullptr );
    
    ASSERT_TRUE( std::fabs( l2->getXLimit() - st::DEFAULT_X_LIMIT ) < 
                 real_limit_t::epsilon() );
    
    ASSERT_TRUE( std::fabs( l2->getYLimit() - st::DEFAULT_Y_LIMIT ) < 
                 real_limit_t::epsilon() );
    
    ASSERT_TRUE( st::ARCH_STATUS_SUCCESS == ::NS(Limit_copy)( 
        limit.getCApiPtr(), l2->getCApiPtr() ) );
    
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values)( 
        limit.getCApiPtr(), l2->getCApiPtr() ) );
        
    be_limit_t* l3 = eb.add<be_limit_t >( real_t{ 2.0 }, real_t{ 3.0 } );
    
    ASSERT_TRUE( l3 != nullptr );
    ASSERT_TRUE( std::fabs( real_t{ 2.0 } - l3->getXLimit() ) < 
                 real_limit_t::epsilon() );
    
    ASSERT_TRUE( std::fabs( real_t{ 3.0 } - l3->getYLimit() ) <
                 real_limit_t::epsilon() );
    
    l3->setXLimit( l3->getYLimit() );
    
    ASSERT_TRUE( std::fabs( l3->getXLimit() - l3->getYLimit() ) < 
                 real_limit_t::epsilon() );
    
    be_limit_t* l4 = eb.addCopy< be_limit_t >( limit );
    
    ASSERT_TRUE( l4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values)( 
        l4->getCApiPtr(), limit.getCApiPtr() ) );
    
    l4->setXLimit( l4->getXLimit() + real_t{ 9e-4 } );
    
    ASSERT_TRUE( 0 != ::NS(Limit_compare_values)( 
        l4->getCApiPtr(), limit.getCApiPtr() ) );
    
    ASSERT_TRUE( 0 != ::NS(Limit_compare_values_with_treshold)( 
        l4->getCApiPtr(), limit.getCApiPtr(), real_limit_t::epsilon() ) );
    
    ASSERT_TRUE( 0 == ::NS(Limit_compare_values_with_treshold)( 
        l4->getCApiPtr(), limit.getCApiPtr(), real_t{ 9e-4 } ) );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_cxx.cpp */
